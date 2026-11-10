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
	"encoding/binary"
	"errors"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type refEntry struct {
	v  []byte
	ts uint64
	hc uint64
}

type refModel map[string]refEntry

func openDiffTestTree(t *testing.T) *TBtree {
	opts := DefaultOptions().
		WithMaxKeySize(64).
		WithMaxValueSize(64).
		WithMaxNodeSize(256).
		WithFlushThld(1_000_000).
		WithSyncThld(1_000_000).
		WithMaxActiveSnapshots(1000).
		WithRenewSnapRootAfter(0).
		WithCacheSize(256)

	tree, err := Open(t.TempDir(), opts)
	require.NoError(t, err)

	return tree
}

// flushedSnapshot forces a flush so the returned snapshot is backed by the
// current tree state. With a large flush threshold and zero renewal period,
// Snapshot() may otherwise reuse the previously committed root.
func flushedSnapshot(t *testing.T, tree *TBtree) *Snapshot {
	_, _, err := tree.Flush()
	require.NoError(t, err)

	snap, err := tree.Snapshot()
	require.NoError(t, err)

	return snap
}

func u32key(v uint32) []byte {
	k := make([]byte, 4)
	binary.BigEndian.PutUint32(k, v)
	return k
}

func u64val(v uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, v)
	return b
}

func snapshotModel(t *testing.T, snap *Snapshot) refModel {
	r, err := snap.NewReader(ReaderSpec{})
	require.NoError(t, err)
	defer func() { require.NoError(t, r.Close()) }()

	m := make(refModel)

	for {
		k, v, ts, hc, err := r.Read()
		if errors.Is(err, ErrNoMoreEntries) {
			break
		}
		require.NoError(t, err)

		m[string(k)] = refEntry{v: cp(v), ts: ts, hc: hc}
	}

	return m
}

func cloneModel(m refModel) refModel {
	c := make(refModel, len(m))
	for k, e := range m {
		c[k] = refEntry{v: cp(e.v), ts: e.ts, hc: e.hc}
	}
	return c
}

func specAdmits(key []byte, spec DiffReaderSpec) bool {
	if len(spec.Prefix) > 0 && !bytes.HasPrefix(key, spec.Prefix) {
		return false
	}

	if len(spec.SeekKey) > 0 {
		cmp := bytes.Compare(key, spec.SeekKey)
		if cmp < 0 || (cmp == 0 && !spec.InclusiveSeek) {
			return false
		}
	}

	if len(spec.EndKey) > 0 {
		cmp := bytes.Compare(key, spec.EndKey)
		if cmp > 0 || (cmp == 0 && !spec.InclusiveEnd) {
			return false
		}
	}

	return true
}

func expectedDiff(modelA, modelB refModel, spec DiffReaderSpec) []DiffEntry {
	keys := make(map[string]struct{}, len(modelA)+len(modelB))
	for k := range modelA {
		keys[k] = struct{}{}
	}
	for k := range modelB {
		keys[k] = struct{}{}
	}

	sorted := make([]string, 0, len(keys))
	for k := range keys {
		if specAdmits([]byte(k), spec) {
			sorted = append(sorted, k)
		}
	}
	sort.Strings(sorted)

	entries := make([]DiffEntry, 0)

	for _, ks := range sorted {
		ea, inA := modelA[ks]
		eb, inB := modelB[ks]
		k := []byte(ks)

		switch {
		case inA && !inB:
			entries = append(entries, DiffEntry{
				Type:     DiffDeleted,
				Key:      cp(k),
				OldValue: cp(ea.v),
				OldTs:    ea.ts,
				OldHc:    ea.hc,
			})
		case !inA && inB:
			entries = append(entries, DiffEntry{
				Type:     DiffAdded,
				Key:      cp(k),
				NewValue: cp(eb.v),
				NewTs:    eb.ts,
				NewHc:    eb.hc,
			})
		default:
			changed := !bytes.Equal(ea.v, eb.v) ||
				(spec.CompareTimestamp && ea.ts != eb.ts) ||
				(spec.CompareHistoryCount && ea.hc != eb.hc)

			if !changed {
				continue
			}

			entries = append(entries, DiffEntry{
				Type:     DiffChanged,
				Key:      cp(k),
				OldValue: cp(ea.v),
				NewValue: cp(eb.v),
				OldTs:    ea.ts,
				NewTs:    eb.ts,
				OldHc:    ea.hc,
				NewHc:    eb.hc,
			})
		}
	}

	return entries
}

func drainDiff(t *testing.T, d *DiffReader) []DiffEntry {
	entries := make([]DiffEntry, 0)
	var prevKey []byte

	for {
		e, err := d.Read()
		if errors.Is(err, ErrNoMoreEntries) {
			break
		}
		require.NoError(t, err)
		require.NotNil(t, e)

		if prevKey != nil {
			require.Equal(t, -1, bytes.Compare(prevKey, e.Key), "entries must be emitted in ascending key order")
		}
		prevKey = cp(e.Key)

		entries = append(entries, *e)
	}

	return entries
}

func requireEntriesEqual(t *testing.T, expected, actual []DiffEntry) {
	require.Equal(t, len(expected), len(actual), "entry count mismatch")

	for i := range expected {
		assert.Equal(t, expected[i].Type, actual[i].Type, "entry %d type", i)
		assert.Equal(t, expected[i].Key, actual[i].Key, "entry %d key", i)
		assert.Equal(t, expected[i].OldValue, actual[i].OldValue, "entry %d old value", i)
		assert.Equal(t, expected[i].NewValue, actual[i].NewValue, "entry %d new value", i)
		assert.Equal(t, expected[i].OldTs, actual[i].OldTs, "entry %d old ts", i)
		assert.Equal(t, expected[i].NewTs, actual[i].NewTs, "entry %d new ts", i)
		assert.Equal(t, expected[i].OldHc, actual[i].OldHc, "entry %d old hc", i)
		assert.Equal(t, expected[i].NewHc, actual[i].NewHc, "entry %d new hc", i)
	}
}

func countNodes(t *testing.T, tree *TBtree, n node) int {
	switch nn := n.(type) {
	case *nodeRef:
		loaded, err := tree.nodeAt(nn.off, false)
		require.NoError(t, err)
		return 1 + countNodes(t, tree, loaded)
	case *innerNode:
		c := 1
		for _, ch := range nn.nodes {
			c += countNodes(t, tree, ch)
		}
		return c
	case *leafNode:
		return 1
	default:
		t.Fatalf("unexpected node type %T", n)
		return 0
	}
}

// countNodeRefs counts persisted node references, i.e. the nodes that would
// have to be materialized (read through the cache/storage) on a full
// traversal. It is the baseline against which DiffStats node reads are
// compared to prove shared subtrees are never opened.
func countNodeRefs(n node) int {
	switch nn := n.(type) {
	case *nodeRef:
		return 1 + countNodeRefs(mustLoadNode(nn))
	case *innerNode:
		c := 0
		for _, ch := range nn.nodes {
			c += countNodeRefs(ch)
		}
		return c
	case *leafNode:
		return 0
	default:
		return 0
	}
}

func mustLoadNode(ref *nodeRef) node {
	loaded, err := ref.t.nodeAt(ref.off, false)
	if err != nil {
		panic(err)
	}
	return loaded
}

func TestDiffReaderRandomized(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	type namedSnap struct {
		snap  *Snapshot
		model refModel
	}

	var snaps []namedSnap

	for round := 0; round < 6; round++ {
		for i := 0; i < 40; i++ {
			k := u32key(rnd.Uint32() % 800)
			v := u64val(rnd.Uint64())
			require.NoError(t, tree.Insert(k, v))
		}

		snap := flushedSnapshot(t, tree)

		snaps = append(snaps, namedSnap{snap: snap, model: snapshotModel(t, snap)})
	}

	// a snapshot carrying uncommitted local Sets (never flushed to the tree)
	locSnap := flushedSnapshot(t, tree)
	locModel := cloneModel(snaps[len(snaps)-1].model)

	for i := 0; i < 15; i++ {
		k := append([]byte{0xEE}, u32key(uint32(1000+i))...)
		v := u64val(uint64(0xEE00 + i))

		require.NoError(t, locSnap.Set(k, v))

		_, ts, hc, gerr := locSnap.Get(k)
		require.NoError(t, gerr)
		locModel[string(k)] = refEntry{v: cp(v), ts: ts, hc: hc}
	}

	snaps = append(snaps, namedSnap{snap: locSnap, model: locModel})

	flagSets := []struct {
		ts bool
		hc bool
	}{
		{false, false},
		{true, false},
		{false, true},
		{true, true},
	}

	rangeSpecs := func() []DiffReaderSpec {
		specs := []DiffReaderSpec{{}}

		// prefix window
		prefix := u32key(rnd.Uint32() % 8)[0:2]
		specs = append(specs, DiffReaderSpec{Prefix: cp(prefix)})

		// half-open / half-closed windows
		lo := u32key(rnd.Uint32() % 800)
		hi := u32key(800 + rnd.Uint32()%200)
		specs = append(specs, DiffReaderSpec{
			SeekKey:       lo,
			InclusiveSeek: rnd.Intn(2) == 0,
			EndKey:        hi,
			InclusiveEnd:  rnd.Intn(2) == 0,
		})

		return specs
	}

	comparePair := func(a, b namedSnap, spec DiffReaderSpec) {
		expected := expectedDiff(a.model, b.model, spec)

		d, err := NewDiffReader(a.snap, b.snap, spec)
		require.NoError(t, err)
		actual := drainDiff(t, d)
		requireEntriesEqual(t, expected, actual)
		require.NoError(t, d.Close())

		// reverse creation must swap additions and deletions
		revSpec := spec
		dr, err := NewDiffReader(b.snap, a.snap, revSpec)
		require.NoError(t, err)
		reverse := drainDiff(t, dr)
		require.NoError(t, dr.Close())

		require.Len(t, reverse, len(expected))

		for i, e := range expected {
			switch e.Type {
			case DiffAdded:
				require.Equal(t, DiffDeleted, reverse[i].Type)
				require.Equal(t, e.Key, reverse[i].Key)
				require.Equal(t, e.NewValue, reverse[i].OldValue)
			case DiffDeleted:
				require.Equal(t, DiffAdded, reverse[i].Type)
				require.Equal(t, e.Key, reverse[i].Key)
				require.Equal(t, e.OldValue, reverse[i].NewValue)
			default:
				require.Equal(t, DiffChanged, reverse[i].Type)
				require.Equal(t, e.Key, reverse[i].Key)
				require.Equal(t, e.OldValue, reverse[i].NewValue)
				require.Equal(t, e.NewValue, reverse[i].OldValue)
			}
		}
	}

	for i, a := range snaps {
		for _, j := range []int{0, i, len(snaps) - 1} {
			if j < i {
				continue
			}
			b := snaps[j]

			for _, flags := range flagSets {
				for _, spec := range rangeSpecs() {
					spec.CompareTimestamp = flags.ts
					spec.CompareHistoryCount = flags.hc
					comparePair(a, b, spec)
				}
			}
		}
	}

	// diffing a snapshot against itself yields no difference. The root is a
	// non-persisted inner node so the initial spine is materialized, but every
	// following shared nodeRef subtree is skipped wholesale.
	d, err := NewDiffReader(snaps[0].snap, snaps[0].snap, DiffReaderSpec{})
	require.NoError(t, err)

	entries := drainDiff(t, d)
	require.Empty(t, entries)

	stats := d.Stats()

	if _, isLeaf := snaps[0].snap.root.(*leafNode); !isLeaf {
		require.GreaterOrEqual(t, stats.SharedSubtreesSkipped, 1)
		require.Less(t, stats.NodesReadA+stats.NodesReadB,
			countNodes(t, tree, snaps[0].snap.root))
	}

	require.NoError(t, d.Close())

	for _, s := range snaps {
		require.NoError(t, s.snap.Close())
	}
}

func TestDiffReaderSharedSubtreeSkipping(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	for i := uint32(0); i < 400; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapA := flushedSnapshot(t, tree)
	modelA := snapshotModel(t, snapA)

	totalA := countNodes(t, tree, snapA.root)
	refsA := countNodeRefs(snapA.root)

	for i := uint32(400); i < 420; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapB := flushedSnapshot(t, tree)
	modelB := snapshotModel(t, snapB)

	totalB := countNodes(t, tree, snapB.root)
	refsB := countNodeRefs(snapB.root)

	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{})
	require.NoError(t, err)

	actual := drainDiff(t, d)
	requireEntriesEqual(t, expectedDiff(modelA, modelB, DiffReaderSpec{}), actual)

	stats := d.Stats()

	require.NoError(t, d.Close())
	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())

	require.Greater(t, stats.SharedSubtreesSkipped, 0,
		"at least one shared immutable subtree must be skipped")
	require.Less(t, stats.NodesReadA+stats.NodesReadB, refsA+refsB,
		"node references inside skipped shared subtrees must never be materialized")
	require.Less(t, stats.NodesReadA+stats.NodesReadB, totalA+totalB)
}

func TestDiffReaderSmallRangeAndPrefix(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	for i := uint32(0); i < 200; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapA := flushedSnapshot(t, tree)
	modelA := snapshotModel(t, snapA)

	for i := uint32(50); i < 150; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i)+1)))
	}
	for i := uint32(200); i < 220; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapB := flushedSnapshot(t, tree)
	modelB := snapshotModel(t, snapB)

	specs := []DiffReaderSpec{
		{
			SeekKey:       u32key(60),
			InclusiveSeek: true,
			EndKey:        u32key(90),
			InclusiveEnd:  false,
		},
		{
			SeekKey:       u32key(60),
			InclusiveSeek: false,
			EndKey:        u32key(90),
			InclusiveEnd:  true,
		},
		{
			Prefix: u32key(100)[:1],
		},
		{
			// a window that contains no key
			SeekKey:       u32key(1000),
			InclusiveSeek: true,
			EndKey:        u32key(2000),
			InclusiveEnd:  true,
		},
		{
			// prefix and explicit bounds combined
			Prefix:        u32key(120)[:2],
			SeekKey:       u32key(120),
			InclusiveSeek: true,
			EndKey:        u32key(140),
			InclusiveEnd:  true,
		},
	}

	for _, spec := range specs {
		d, err := NewDiffReader(snapA, snapB, spec)
		require.NoError(t, err)

		requireEntriesEqual(t, expectedDiff(modelA, modelB, spec), drainDiff(t, d))
		require.NoError(t, d.Close())
	}

	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func TestDiffReaderVersionFlags(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	key := u32key(42)
	val := u64val(7)

	require.NoError(t, tree.Insert(key, val))

	snapA := flushedSnapshot(t, tree)
	modelA := snapshotModel(t, snapA)

	// same value bytes, new version: timestamp and history counter change
	require.NoError(t, tree.BulkInsert([]*KVT{{
		K: key,
		V: cp(val),
		T: tree.Ts() + 1,
	}}))

	snapB := flushedSnapshot(t, tree)
	modelB := snapshotModel(t, snapB)

	// with value-only comparison the key is not reported as changed
	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{})
	require.NoError(t, err)
	require.Empty(t, drainDiff(t, d))
	require.NoError(t, d.Close())

	d, err = NewDiffReader(snapA, snapB, DiffReaderSpec{CompareTimestamp: true})
	require.NoError(t, err)
	entries := drainDiff(t, d)
	requireEntriesEqual(t, expectedDiff(modelA, modelB, DiffReaderSpec{CompareTimestamp: true}), entries)
	require.Len(t, entries, 1)
	require.Equal(t, DiffChanged, entries[0].Type)
	require.Equal(t, key, entries[0].Key)
	require.Equal(t, val, entries[0].OldValue)
	require.Equal(t, val, entries[0].NewValue)
	require.NotEqual(t, entries[0].OldTs, entries[0].NewTs)
	require.NoError(t, d.Close())

	d, err = NewDiffReader(snapA, snapB, DiffReaderSpec{CompareHistoryCount: true})
	require.NoError(t, err)
	entries = drainDiff(t, d)
	require.Len(t, entries, 1)
	require.NotEqual(t, entries[0].OldHc, entries[0].NewHc)
	require.NoError(t, d.Close())

	// a one-byte value difference must always be detected: comparisons use
	// the full raw bytes, never short hashes
	require.NoError(t, tree.Insert(key, u64val(8)))

	snapC := flushedSnapshot(t, tree)
	modelC := snapshotModel(t, snapC)

	d, err = NewDiffReader(snapB, snapC, DiffReaderSpec{})
	require.NoError(t, err)
	entries = drainDiff(t, d)
	requireEntriesEqual(t, expectedDiff(modelB, modelC, DiffReaderSpec{}), entries)
	require.Len(t, entries, 1)
	require.Equal(t, DiffChanged, entries[0].Type)
	require.False(t, bytes.Equal(entries[0].OldValue, entries[0].NewValue))
	require.NoError(t, d.Close())

	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
	require.NoError(t, snapC.Close())
}

func TestDiffReaderSameImmutableNodePredicate(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	other := openDiffTestTree(t)
	defer func() { require.NoError(t, other.Close()) }()

	d, err := NewDiffReader(tree.newSnapshot(1, &leafNode{t: tree}), tree.newSnapshot(2, &leafNode{t: tree}), DiffReaderSpec{})
	require.NoError(t, err)
	defer func() { require.NoError(t, d.Close()) }()

	minKey := []byte{1, 2, 3}

	refA := &nodeRef{t: tree, _minKey: minKey, _ts: 10, off: 100, _minOff: 50}
	refAClone := &nodeRef{t: tree, _minKey: cp(minKey), _ts: 10, off: 100, _minOff: 50}
	refOtherGen := &nodeRef{t: other, _minKey: minKey, _ts: 10, off: 100, _minOff: 50}
	refOtherOff := &nodeRef{t: tree, _minKey: minKey, _ts: 10, off: 101, _minOff: 50}
	refOtherTs := &nodeRef{t: tree, _minKey: minKey, _ts: 11, off: 100, _minOff: 50}

	require.True(t, d.sameImmutableNode(refA, refAClone))
	require.True(t, d.sameImmutableNode(refA, refA))

	// equal numeric offset is not enough across storage generations
	require.False(t, d.sameImmutableNode(refA, refOtherGen))
	require.False(t, d.sameImmutableNode(refA, refOtherOff))
	require.False(t, d.sameImmutableNode(refA, refOtherTs))

	// equal pointers to mutable in-memory nodes never count as shared
	leaf := &leafNode{t: tree, mut: true}
	inner := &innerNode{t: tree, mut: true}
	require.False(t, d.sameImmutableNode(leaf, leaf))
	require.False(t, d.sameImmutableNode(inner, inner))
	require.False(t, d.sameImmutableNode(nil, refA))
	require.False(t, d.sameImmutableNode(refA, nil))

	require.NotEqual(t, tree.storageGen, other.storageGen)
}

func TestDiffReaderIllegalArguments(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	snapA := flushedSnapshot(t, tree)
	snapB := flushedSnapshot(t, tree)

	_, err := NewDiffReader(nil, snapB, DiffReaderSpec{})
	require.ErrorIs(t, err, ErrIllegalArguments)

	_, err = NewDiffReader(snapA, nil, DiffReaderSpec{})
	require.ErrorIs(t, err, ErrIllegalArguments)

	other := openDiffTestTree(t)
	defer func() { require.NoError(t, other.Close()) }()

	otherSnap, err := other.Snapshot()
	require.NoError(t, err)

	_, err = NewDiffReader(snapA, otherSnap, DiffReaderSpec{})
	require.ErrorIs(t, err, ErrIllegalArguments)
	require.ErrorIs(t, err, ErrDifferentOrigin)
	require.NoError(t, otherSnap.Close())

	_, err = NewDiffReader(snapA, snapB, DiffReaderSpec{Limit: -1})
	require.ErrorIs(t, err, ErrIllegalArguments)

	_, err = NewDiffReader(snapA, snapB, DiffReaderSpec{Prefix: make([]byte, tree.maxKeySize+1)})
	require.ErrorIs(t, err, ErrIllegalArguments)

	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func TestDiffReaderPinnedRootsAndLifecycle(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	for i := uint32(0); i < 100; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapA := flushedSnapshot(t, tree)
	modelA := snapshotModel(t, snapA)

	for i := uint32(100); i < 120; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapB := flushedSnapshot(t, tree)
	modelB := snapshotModel(t, snapB)

	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{})
	require.NoError(t, err)

	// neither input snapshot can be closed while the diff is active
	require.ErrorIs(t, snapA.Close(), ErrReadersNotClosed)
	require.ErrorIs(t, snapB.Close(), ErrReadersNotClosed)

	// concurrent-looking Sets swapping snapshot roots must not leak into
	// the pinned diff
	for i := uint32(500); i < 520; i++ {
		require.NoError(t, snapA.Set(u32key(i), u64val(i)))
		require.NoError(t, snapB.Set(u32key(i), u64val(i+1)))
	}

	requireEntriesEqual(t, expectedDiff(modelA, modelB, DiffReaderSpec{}), drainDiff(t, d))
	require.NoError(t, d.Close())

	// pins released: no reader leak and snapshots close normally
	require.Empty(t, snapA.readers)
	require.Empty(t, snapB.readers)

	_, err = d.Read()
	require.ErrorIs(t, err, ErrAlreadyClosed)
	require.ErrorIs(t, d.Close(), ErrAlreadyClosed)

	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func TestDiffReaderOwnedReturnValues(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	require.NoError(t, tree.Insert(u32key(1), u64val(1)))
	snapA := flushedSnapshot(t, tree)

	require.NoError(t, tree.Insert(u32key(2), u64val(2)))
	snapB := flushedSnapshot(t, tree)

	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{})
	require.NoError(t, err)

	first, err := d.Read()
	require.NoError(t, err)

	second, err := d.Read()
	require.NoError(t, err)

	// returned slices are independent allocations
	require.NotSame(t, first.Key, second.Key)

	// mutating a previously returned entry cannot corrupt the next one
	for i := range first.Key {
		first.Key[i] = 0xFF
	}
	for i := range first.NewValue {
		first.NewValue[i] = 0xFF
	}

	require.Equal(t, u32key(2), second.Key)
	require.Equal(t, u64val(2), second.NewValue)

	require.NoError(t, d.Close())
	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func TestDiffReaderLimit(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	snapA := flushedSnapshot(t, tree)

	for i := uint32(0); i < 30; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapB := flushedSnapshot(t, tree)
	modelB := snapshotModel(t, snapB)

	const limit = 5

	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{Limit: limit})
	require.NoError(t, err)

	entries := drainDiff(t, d)
	require.Len(t, entries, limit)

	// limited output is the prefix of the full diff in key order
	full, err := NewDiffReader(snapA, snapB, DiffReaderSpec{})
	require.NoError(t, err)
	expected := drainDiff(t, full)
	require.NoError(t, full.Close())
	require.Len(t, expected, len(modelB))
	requireEntriesEqual(t, expected[:limit], entries)
	require.NoError(t, d.Close())

	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func TestDiffReaderCorruptNodeFails(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	for i := uint32(0); i < 20; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapGood := flushedSnapshot(t, tree)

	// a root nodeRef pointing past the nodes log end
	bogusRoot := &nodeRef{
		t:       tree,
		_minKey: []byte{0},
		_ts:     1,
		off:     tree.committedNLogSize + 4096,
		_minOff: tree.committedNLogSize + 4096,
	}
	snapBogus := tree.newSnapshot(999999, bogusRoot)

	d, err := NewDiffReader(snapGood, snapBogus, DiffReaderSpec{})
	require.NoError(t, err)

	_, err = d.Read()
	require.Error(t, err)
	require.False(t, errors.Is(err, ErrNoMoreEntries),
		"corrupt nodes must surface as failures, not a premature end-of-stream")
	require.ErrorIs(t, err, ErrReadingFileContent)

	// failure is sticky and never turns into EOF
	_, err = d.Read()
	require.Error(t, err)
	require.ErrorIs(t, err, ErrReadingFileContent)

	require.NoError(t, d.Close())
	require.NoError(t, snapGood.Close())
	require.NoError(t, snapBogus.Close())
}

func TestDiffReaderCancellation(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	snapA := flushedSnapshot(t, tree)

	for i := uint32(0); i < 100; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapB := flushedSnapshot(t, tree)

	ctx, cancel := context.WithCancel(context.Background())

	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{Ctx: ctx})
	require.NoError(t, err)

	_, err = d.Read()
	require.NoError(t, err)

	cancel()

	_, err = d.Read()
	require.ErrorIs(t, err, context.Canceled)

	// cancellation released both pins
	require.Empty(t, snapA.readers)
	require.Empty(t, snapB.readers)

	_, err = d.Read()
	require.ErrorIs(t, err, ErrAlreadyClosed)
	require.ErrorIs(t, d.Close(), ErrAlreadyClosed)

	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func TestDiffReaderConcurrentSetCloseAndFlush(t *testing.T) {
	tree := openDiffTestTree(t)
	defer func() { require.NoError(t, tree.Close()) }()

	for i := uint32(0); i < 200; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapA := flushedSnapshot(t, tree)
	modelA := snapshotModel(t, snapA)

	for i := uint32(200); i < 260; i++ {
		require.NoError(t, tree.Insert(u32key(i), u64val(uint64(i))))
	}

	snapB := flushedSnapshot(t, tree)
	modelB := snapshotModel(t, snapB)

	d, err := NewDiffReader(snapA, snapB, DiffReaderSpec{})
	require.NoError(t, err)

	var wg sync.WaitGroup
	errCh := make(chan error, 16)

	// writers swapping roots of both snapshots while the diff is iterating
	for w := 0; w < 2; w++ {
		wg.Add(1)

		go func(snap *Snapshot, base uint32) {
			defer wg.Done()

			for i := uint32(0); i < 100; i++ {
				if err := snap.Set(u32key(base+i), u64val(uint64(base+i))); err != nil {
					errCh <- err
					return
				}
			}
		}(pickSnap(w, snapA, snapB), uint32(1000+w*1000))
	}

	// close attempts must fail while the diff reader is active
	wg.Add(1)

	go func() {
		defer wg.Done()

		for i := 0; i < 50; i++ {
			if err := snapA.Close(); !errors.Is(err, ErrReadersNotClosed) {
				errCh <- err
				return
			}
			if err := snapB.Close(); !errors.Is(err, ErrReadersNotClosed) {
				errCh <- err
				return
			}
		}
	}()

	// flushes move the tree storage forward while the diff pins old roots
	wg.Add(1)

	go func() {
		defer wg.Done()

		for i := 0; i < 5; i++ {
			if err := tree.Insert(u32key(uint32(9000+i)), u64val(uint64(i))); err != nil {
				errCh <- err
				return
			}
			if _, _, err := tree.Flush(); err != nil {
				errCh <- err
				return
			}
		}
	}()

	entries := drainDiff(t, d)

	wg.Wait()
	close(errCh)

	for err := range errCh {
		require.NoError(t, err)
	}

	// the diff reflects the pinned roots only, never the concurrent Sets
	requireEntriesEqual(t, expectedDiff(modelA, modelB, DiffReaderSpec{}), entries)

	require.NoError(t, d.Close())
	require.Empty(t, snapA.readers)
	require.Empty(t, snapB.readers)
	require.NoError(t, snapA.Close())
	require.NoError(t, snapB.Close())
}

func pickSnap(i int, a, b *Snapshot) *Snapshot {
	if i == 0 {
		return a
	}
	return b
}
