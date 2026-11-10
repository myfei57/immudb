/*
Copyright 2026 Codenotary Inc. All rights reserved.

SPDX-License-Identifier: BUSL-1.1
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://mariadb.com/bsl11/

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tbtree

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
)

// ErrDifferentOrigin is returned when attempting to diff snapshots that do
// not belong to the same TBtree instance. Subtree sharing can only be proven
// among snapshots of the same storage generation.
var ErrDifferentOrigin = errors.New("tbtree: snapshots belong to different tbtree instances")

// DiffEntryType identifies the kind of difference reported by a DiffReader.
type DiffEntryType uint8

const (
	// DiffAdded is reported for keys that only exist in the second (newer)
	// snapshot.
	DiffAdded DiffEntryType = iota
	// DiffDeleted is reported for keys that only exist in the first (older)
	// snapshot.
	DiffDeleted
	// DiffChanged is reported for keys existing in both snapshots whose
	// current value bytes (or explicitly selected version metadata) differ.
	DiffChanged
)

// DiffEntry is a single difference between two snapshots.
//
// Every byte slice returned in a DiffEntry is owned by the caller: entries
// are copied out of the tree nodes, so advancing the reader never mutates
// previously returned entries.
//
// For DiffAdded entries Old* fields are zero-valued, for DiffDeleted entries
// New* fields are zero-valued.
type DiffEntry struct {
	Type DiffEntryType

	Key []byte

	OldValue []byte
	NewValue []byte

	OldTs uint64
	NewTs uint64

	OldHc uint64
	NewHc uint64
}

// DiffReaderSpec configures a DiffReader.
type DiffReaderSpec struct {
	// Prefix restricts the diff to keys having this prefix. An empty prefix
	// covers the whole key space.
	Prefix []byte

	// SeekKey positions the reader at the first key greater than (or equal
	// to, when InclusiveSeek is true) SeekKey. When empty or smaller than
	// Prefix, the prefix lower bound is used as an inclusive seek key.
	SeekKey       []byte
	InclusiveSeek bool

	// EndKey limits the diff to keys smaller than (or equal to, when
	// InclusiveEnd is true) EndKey. When empty or beyond the prefix upper
	// bound, the greatest key matching the prefix is used inclusively.
	EndKey       []byte
	InclusiveEnd bool

	// Limit caps the number of entries emitted. A value of zero means no
	// limit. The limit only controls how many entries are produced: it never
	// masks read or corruption errors, which are always returned as errors
	// rather than turned into an early end-of-stream.
	Limit int

	// CompareTimestamp, when true, makes a difference of the latest entry
	// timestamp (with equal value bytes) a DiffChanged entry.
	CompareTimestamp bool

	// CompareHistoryCount, when true, makes a difference of the history
	// counter (with equal value bytes) a DiffChanged entry.
	CompareHistoryCount bool

	// Ctx cancels an in-progress iteration. Cancelling the context releases
	// the pins held on both input snapshots. A nil context is replaced by
	// context.Background().
	Ctx context.Context
}

// DiffStats exposes node-level counters collected while producing a diff.
// They are intended to prove that whole shared immutable subtrees are
// skipped instead of being traversed.
type DiffStats struct {
	// NodesReadA / NodesReadB count the node materializations (nodeRef
	// resolutions through the node cache/storage) performed on each side.
	// Nodes inside a skipped shared subtree are never counted.
	NodesReadA int
	NodesReadB int

	// SharedSubtreesSkipped counts how many identical immutable subtrees
	// were skipped wholesale on both sides.
	SharedSubtreesSkipped int

	Added   int
	Deleted int
	Changed int
}

const (
	sideA = iota
	sideB
)

const (
	keyBefore = -1
	keyWithin = 0
	keyAfter  = 1
)

// diffFrame is one level of the per-side descent stack. idx always points at
// the next child to process.
type diffFrame struct {
	inner *innerNode
	idx   int
}

// diffSide is a lazy, range-bounded cursor over one pinned snapshot root.
type diffSide struct {
	stack []diffFrame
	leaf  *leafNode
	lidx  int

	positioned bool
	finished   bool
}

func (s *diffSide) needsEntry() bool {
	return !s.finished && (s.leaf == nil || s.lidx >= len(s.leaf.values))
}

func (s *diffSide) markFinished() {
	s.finished = true
	s.leaf = nil
	s.stack = nil
}

// DiffReader lazily emits the differences between two same-origin snapshots
// in key byte order, without ever building a complete key map.
//
// The roots of both snapshots are pinned at construction time: subsequent
// Snapshot.Set calls (which swap the snapshot root through copy-on-write)
// can never mix a new root into an in-progress diff. The reader registers
// itself on both snapshots, so neither input snapshot can be closed while
// the diff is active (Snapshot.Close returns ErrReadersNotClosed). Closing
// the DiffReader never closes the input snapshots.
type DiffReader struct {
	snapA *Snapshot
	snapB *Snapshot

	idA int
	idB int

	rootA node
	rootB node

	// lockLow/lockHigh impose one global acquisition order (by snapshot
	// address) so reverse-created readers over the same snapshot pair cannot
	// deadlock against each other.
	lockLow  *Snapshot
	lockHigh *Snapshot

	ctx context.Context

	prefix        []byte
	seekKey       []byte
	endKey        []byte
	inclusiveSeek bool
	inclusiveEnd  bool

	compareTs bool
	compareHc bool

	limit   int
	emitted int

	sideA diffSide
	sideB diffSide

	started   bool
	exhausted bool
	closed    bool
	err       error

	stats DiffStats

	mu sync.Mutex
}

// NewDiffReader creates a DiffReader over two snapshots of the same TBtree
// instance. snapA is treated as the older/baseline state and snapB as the
// newer state, so keys only in snapB are additions and keys only in snapA
// are deletions. Reversing the arguments swaps additions and deletions.
func NewDiffReader(snapA, snapB *Snapshot, spec DiffReaderSpec) (*DiffReader, error) {
	if snapA == nil || snapB == nil {
		return nil, fmt.Errorf("%w: snapshots cannot be nil", ErrIllegalArguments)
	}

	// Same-origin is mandatory: node offset identity is only meaningful
	// within one TBtree's append-only nodes log generation.
	if snapA.t != snapB.t {
		return nil, fmt.Errorf("%w: %w", ErrIllegalArguments, ErrDifferentOrigin)
	}

	if spec.Limit < 0 {
		return nil, fmt.Errorf("%w: limit cannot be negative", ErrIllegalArguments)
	}

	validate := func(s *Snapshot) error {
		s.mutex.RLock()
		defer s.mutex.RUnlock()

		if s.closed {
			return ErrAlreadyClosed
		}

		if len(spec.Prefix) > s.t.maxKeySize ||
			len(spec.SeekKey) > s.t.maxKeySize ||
			len(spec.EndKey) > s.t.maxKeySize {
			return ErrIllegalArguments
		}

		return nil
	}

	if err := validate(snapA); err != nil {
		return nil, err
	}

	if err := validate(snapB); err != nil {
		return nil, err
	}

	ctx := spec.Ctx
	if ctx == nil {
		ctx = context.Background()
	}

	// Normalize the key window. Every bound is copied so the caller retains
	// ownership of the spec's slices.
	prefix := cp(spec.Prefix)

	seekKey := cp(spec.SeekKey)
	inclusiveSeek := spec.InclusiveSeek
	if len(seekKey) == 0 || bytes.Compare(seekKey, prefix) < 0 {
		seekKey = cp(prefix)
		inclusiveSeek = true
	}

	greatestPrefixedKey := greatestKeyOfSize(snapA.t.maxKeySize)
	copy(greatestPrefixedKey, prefix)

	endKey := cp(spec.EndKey)
	inclusiveEnd := spec.InclusiveEnd
	if len(endKey) == 0 || bytes.Compare(endKey, greatestPrefixedKey) > 0 {
		endKey = greatestPrefixedKey
		inclusiveEnd = true
	}

	d := &DiffReader{
		snapA: snapA,
		snapB: snapB,

		ctx: ctx,

		prefix:        prefix,
		seekKey:       seekKey,
		endKey:        endKey,
		inclusiveSeek: inclusiveSeek,
		inclusiveEnd:  inclusiveEnd,

		compareTs: spec.CompareTimestamp,
		compareHc: spec.CompareHistoryCount,

		limit: spec.Limit,
	}

	d.lockLow, d.lockHigh = snapA, snapB
	if reflect.ValueOf(snapB).Pointer() < reflect.ValueOf(snapA).Pointer() {
		d.lockLow, d.lockHigh = snapB, snapA
	}

	// Pin both snapshots. Registrations are performed one snapshot at a time
	// (nested snapshot locks are never taken here), and a failure while
	// registering the second snapshot rolls the first registration back.
	register := func(s *Snapshot) (int, node, error) {
		s.mutex.Lock()
		defer s.mutex.Unlock()

		if s.closed {
			return 0, nil, ErrAlreadyClosed
		}

		id := s.maxReaderID
		s.maxReaderID++
		s.readers[id] = d

		return id, s.root, nil
	}

	idA, rootA, err := register(snapA)
	if err != nil {
		return nil, err
	}

	idB, rootB, err := register(snapB)
	if err != nil {
		snapA.mutex.Lock()
		delete(snapA.readers, idA)
		snapA.mutex.Unlock()
		return nil, err
	}

	// Re-pin the roots observed under the registration lock. These are the
	// only roots the diff will ever traverse.
	d.idA, d.idB = idA, idB
	d.rootA, d.rootB = rootA, rootB

	return d, nil
}

// Read returns the next difference in key byte order, or ErrNoMoreEntries
// when the diff is complete. Read or corruption errors are returned as
// errors and stick to the reader, never as a premature end-of-stream.
func (d *DiffReader) Read() (entry *DiffEntry, err error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil, ErrAlreadyClosed
	}

	if d.err != nil {
		return nil, d.err
	}

	if err = d.ctx.Err(); err != nil {
		// cancellation releases both pins immediately
		d.releaseLocked()
		return nil, err
	}

	if d.exhausted {
		return nil, ErrNoMoreEntries
	}

	// Short critical sections: the snapshot locks only guard the pinned node
	// graphs against in-place mutation of mutable (uncommitted) nodes. The
	// traversal itself uses node methods and t.nodeAt directly; no public
	// Snapshot/Reader method (which would re-acquire these locks) is ever
	// called while holding them, so a waiting writer cannot deadlock us.
	sameSnapshot := d.lockLow == d.lockHigh

	d.lockLow.mutex.RLock()
	if !sameSnapshot {
		d.lockHigh.mutex.RLock()
	}

	entry, err = d.nextEntryLocked()

	if !sameSnapshot {
		d.lockHigh.mutex.RUnlock()
	}
	d.lockLow.mutex.RUnlock()

	if err != nil {
		// context cancellation takes precedence and releases the pins
		if ctxErr := d.ctx.Err(); ctxErr != nil {
			d.releaseLocked()
			return nil, ctxErr
		}

		d.err = err
		return nil, err
	}

	if entry == nil {
		d.exhausted = true
		return nil, ErrNoMoreEntries
	}

	d.emitted++

	switch entry.Type {
	case DiffAdded:
		d.stats.Added++
	case DiffDeleted:
		d.stats.Deleted++
	case DiffChanged:
		d.stats.Changed++
	}

	// The limit restricts output only: no traversal state or integrity
	// status is rewritten to reach it.
	if d.limit > 0 && d.emitted >= d.limit {
		d.exhausted = true
	}

	return entry, nil
}

// Stats returns the node read and emission counters collected so far.
func (d *DiffReader) Stats() DiffStats {
	d.mu.Lock()
	defer d.mu.Unlock()

	return d.stats
}

// Close releases the pins held on both input snapshots. It does not close
// the input snapshots themselves. While a DiffReader is open, closing either
// input snapshot fails with ErrReadersNotClosed.
func (d *DiffReader) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	return d.releaseLocked()
}

func (d *DiffReader) releaseLocked() error {
	if d.closed {
		return ErrAlreadyClosed
	}

	d.snapA.mutex.Lock()
	delete(d.snapA.readers, d.idA)
	d.snapA.mutex.Unlock()

	d.snapB.mutex.Lock()
	delete(d.snapB.readers, d.idB)
	d.snapB.mutex.Unlock()

	d.closed = true
	d.sideA.markFinished()
	d.sideB.markFinished()

	return nil
}

func (d *DiffReader) side(which int) *diffSide {
	if which == sideA {
		return &d.sideA
	}
	return &d.sideB
}

// sameImmutableNode proves that two nodes denote the same immutable persisted
// subtree of the same storage generation.
//
// Equality is accepted only when both nodes are nodeRefs (mutable in-memory
// inner/leaf nodes never qualify, even when pointer-equal) of the same TBtree
// generation with the same persisted offset and matching ref metadata. Equal
// numeric offsets alone are never enough: offsets restart from zero across
// reopen/compaction generations.
func (d *DiffReader) sameImmutableNode(a, b node) bool {
	ra, okA := a.(*nodeRef)
	rb, okB := b.(*nodeRef)

	if !okA || !okB || ra == nil || rb == nil {
		return false
	}

	if ra.t != rb.t || ra.t.storageGen != rb.t.storageGen {
		return false
	}

	if ra.off != rb.off || ra._minOff != rb._minOff || ra._ts != rb._ts {
		return false
	}

	return bytes.Equal(ra._minKey, rb._minKey)
}

func (d *DiffReader) materialize(which int, n node) (node, error) {
	ref, ok := n.(*nodeRef)
	if !ok {
		if n == nil {
			return nil, fmt.Errorf("%w: nil node while traversing snapshot", ErrReadingFileContent)
		}
		return n, nil
	}

	if which == sideA {
		d.stats.NodesReadA++
	} else {
		d.stats.NodesReadB++
	}

	loaded, err := ref.t.nodeAt(ref.off, false)
	if err != nil {
		// A truncated/corrupt node or any storage error is a failure, never
		// an end-of-stream.
		return nil, fmt.Errorf("%w: unable to read node at offset %d: %v", ErrReadingFileContent, ref.off, err)
	}

	return loaded, nil
}

func (d *DiffReader) corruptNodeError(n node) error {
	return fmt.Errorf("%w: unexpected node type %T while traversing snapshot", ErrReadingFileContent, n)
}

// seek positions a side at the first leaf entry not before the seek key.
func (s *diffSide) seek(d *DiffReader, which int, root node) error {
	s.positioned = true

	n := root

	for {
		m, err := d.materialize(which, n)
		if err != nil {
			return err
		}

		switch nn := m.(type) {
		case *innerNode:
			if len(nn.nodes) == 0 {
				return d.corruptNodeError(m)
			}

			idx := 0
			if len(d.seekKey) > 0 {
				idx = nn.indexOf(d.seekKey)
			}

			s.stack = append(s.stack, diffFrame{inner: nn, idx: idx})
			n = nn.nodes[idx]

		case *leafNode:
			s.leaf = nn
			s.lidx = 0

			if len(d.seekKey) > 0 {
				i, _ := nn.indexOf(d.seekKey)

				if !d.inclusiveSeek && i < len(nn.values) && bytes.Equal(nn.values[i].key, d.seekKey) {
					i++
				}

				s.lidx = i
			}

			return nil

		default:
			return d.corruptNodeError(m)
		}
	}
}

// descendLeftmost opens a subtree that was selected by peekNextChild.
func (s *diffSide) descendLeftmost(d *DiffReader, which int, n node) error {
	for {
		m, err := d.materialize(which, n)
		if err != nil {
			return err
		}

		switch nn := m.(type) {
		case *innerNode:
			if len(nn.nodes) == 0 {
				return d.corruptNodeError(m)
			}

			s.stack = append(s.stack, diffFrame{inner: nn, idx: 0})
			n = nn.nodes[0]

		case *leafNode:
			s.leaf = nn
			s.lidx = 0
			return nil

		default:
			return d.corruptNodeError(m)
		}
	}
}

// peekNextChild advances the exhausted frame stack and returns the next
// subtree to descend, or nil when the side is fully consumed.
func (s *diffSide) peekNextChild() node {
	for len(s.stack) > 0 {
		top := &s.stack[len(s.stack)-1]
		top.idx++

		if top.idx < len(top.inner.nodes) {
			return top.inner.nodes[top.idx]
		}

		s.stack = s.stack[:len(s.stack)-1]
	}

	s.finished = true
	return nil
}

func (d *DiffReader) advanceSide(which int) error {
	s := d.side(which)

	if s.leaf != nil {
		// The current leaf is fully consumed; dropping it keeps the side
		// eligible for a coordinated shared-subtree skip on the next round.
		s.leaf = nil
		return nil
	}

	child := s.peekNextChild()
	if child == nil {
		return nil
	}

	return s.descendLeftmost(d, which, child)
}

// resolveNeedy makes one bounded advancement step on the sides that do not
// have a current entry.
func (d *DiffReader) resolveNeedy() error {
	needyA := d.sideA.needsEntry()
	needyB := d.sideB.needsEntry()

	// Coordinated fast path: both sides are between subtrees. When their
	// next children are the same immutable persisted subtree, the whole
	// subtree is skipped on both sides without reading any of its nodes.
	if needyA && needyB && d.sideA.leaf == nil && d.sideB.leaf == nil {

		childA := d.sideA.peekNextChild()
		childB := d.sideB.peekNextChild()

		if childA != nil && childB != nil && d.sameImmutableNode(childA, childB) {
			d.stats.SharedSubtreesSkipped++
			return nil
		}

		if childA != nil {
			if err := d.sideA.descendLeftmost(d, sideA, childA); err != nil {
				return err
			}
		}

		if childB != nil {
			if err := d.sideB.descendLeftmost(d, sideB, childB); err != nil {
				return err
			}
		}

		return nil
	}

	if needyA {
		if err := d.advanceSide(sideA); err != nil {
			return err
		}
	}

	if needyB {
		if err := d.advanceSide(sideB); err != nil {
			return err
		}
	}

	return nil
}

func (d *DiffReader) keyStatus(key []byte) int {
	if !d.inclusiveSeek && bytes.Equal(key, d.seekKey) {
		return keyBefore
	}

	if len(d.prefix) > 0 &&
		(len(key) < len(d.prefix) || !bytes.Equal(key[:len(d.prefix)], d.prefix)) {
		// keys are visited in ascending order and the seek key is never
		// smaller than the prefix, so a mismatch means the prefix window has
		// been left entirely
		return keyAfter
	}

	cmp := bytes.Compare(key, d.endKey)
	if cmp > 0 || (cmp == 0 && !d.inclusiveEnd) {
		return keyAfter
	}

	return keyWithin
}

func (d *DiffReader) nextEntryLocked() (*DiffEntry, error) {
	if !d.started {
		d.started = true

		// Identical persisted roots make the whole trees identical; no node
		// is read at all.
		if d.sameImmutableNode(d.rootA, d.rootB) {
			d.stats.SharedSubtreesSkipped++
			return nil, nil
		}
	}

	if !d.sideA.positioned {
		if err := d.sideA.seek(d, sideA, d.rootA); err != nil {
			return nil, err
		}
	}

	if !d.sideB.positioned {
		if err := d.sideB.seek(d, sideB, d.rootB); err != nil {
			return nil, err
		}
	}

	for {
		if err := d.ctx.Err(); err != nil {
			return nil, err
		}

		for d.sideA.needsEntry() || d.sideB.needsEntry() {
			if err := d.ctx.Err(); err != nil {
				return nil, err
			}

			if err := d.resolveNeedy(); err != nil {
				return nil, err
			}
		}

		if d.sideA.finished && d.sideB.finished {
			return nil, nil
		}

		if d.sideA.finished {
			if entry, err := d.emitSingle(sideB, DiffAdded); err != nil || entry != nil {
				return entry, err
			}
			continue
		}

		if d.sideB.finished {
			if entry, err := d.emitSingle(sideA, DiffDeleted); err != nil || entry != nil {
				return entry, err
			}
			continue
		}

		entry, err := d.compareCurrent()
		if err != nil {
			return nil, err
		}

		if entry != nil {
			return entry, nil
		}
	}
}

// compareCurrent classifies the entries currently pointed at on both sides.
// A nil entry with no error means the sides advanced without producing a
// difference and the caller must continue the loop.
func (d *DiffReader) compareCurrent() (*DiffEntry, error) {
	lvA := d.sideA.leaf.values[d.sideA.lidx]
	lvB := d.sideB.leaf.values[d.sideB.lidx]

	statusA := d.keyStatus(lvA.key)
	statusB := d.keyStatus(lvB.key)

	if statusA == keyAfter {
		d.sideA.markFinished()
		return nil, nil
	}

	if statusB == keyAfter {
		d.sideB.markFinished()
		return nil, nil
	}

	if statusA == keyBefore {
		d.sideA.lidx++
		return nil, nil
	}

	if statusB == keyBefore {
		d.sideB.lidx++
		return nil, nil
	}

	cmp := bytes.Compare(lvA.key, lvB.key)

	switch {
	case cmp < 0:
		d.sideA.lidx++
		return d.deletedEntry(lvA), nil

	case cmp > 0:
		d.sideB.lidx++
		return d.addedEntry(lvB), nil
	}

	// Same key: compare the full current value bytes plus only the
	// explicitly selected version metadata. No short hashes are used, so
	// hash collisions can never hide a change.
	tvA := lvA.timedValue()
	tvB := lvB.timedValue()
	hcA := lvA.historyCount()
	hcB := lvB.historyCount()

	changed := !bytes.Equal(tvA.Value, tvB.Value) ||
		(d.compareTs && tvA.Ts != tvB.Ts) ||
		(d.compareHc && hcA != hcB)

	d.sideA.lidx++
	d.sideB.lidx++

	if !changed {
		return nil, nil
	}

	return d.changedEntry(lvA.key, tvA.Value, tvA.Ts, hcA, tvB.Value, tvB.Ts, hcB), nil
}

func (d *DiffReader) emitSingle(which int, typ DiffEntryType) (*DiffEntry, error) {
	s := d.side(which)
	lv := s.leaf.values[s.lidx]

	switch d.keyStatus(lv.key) {
	case keyAfter:
		s.markFinished()
		return nil, nil
	case keyBefore:
		s.lidx++
		return nil, nil
	}

	s.lidx++

	if typ == DiffAdded {
		return d.addedEntry(lv), nil
	}

	return d.deletedEntry(lv), nil
}

func (d *DiffReader) addedEntry(lv *leafValue) *DiffEntry {
	tv := lv.timedValue()

	return &DiffEntry{
		Type:     DiffAdded,
		Key:      cp(lv.key),
		NewValue: cp(tv.Value),
		NewTs:    tv.Ts,
		NewHc:    lv.historyCount(),
	}
}

func (d *DiffReader) deletedEntry(lv *leafValue) *DiffEntry {
	tv := lv.timedValue()

	return &DiffEntry{
		Type:     DiffDeleted,
		Key:      cp(lv.key),
		OldValue: cp(tv.Value),
		OldTs:    tv.Ts,
		OldHc:    lv.historyCount(),
	}
}

func (d *DiffReader) changedEntry(
	key []byte,
	oldValue []byte,
	oldTs uint64,
	oldHc uint64,
	newValue []byte,
	newTs uint64,
	newHc uint64,
) *DiffEntry {
	return &DiffEntry{
		Type:     DiffChanged,
		Key:      cp(key),
		OldValue: cp(oldValue),
		NewValue: cp(newValue),
		OldTs:    oldTs,
		NewTs:    newTs,
		OldHc:    oldHc,
		NewHc:    newHc,
	}
}
