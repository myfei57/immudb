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

package store

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
)

// ErrMergedKeyReaderInconsistentEntries is returned when the same entry is
// observed through different ranges of a merged key reader but the observed
// contents do not match. The merged reader never picks one copy arbitrarily.
var ErrMergedKeyReaderInconsistentEntries = errors.New("merged key reader: inconsistent entries observed for the same key version")

// NewMergedKeyReader returns a KeyReader that yields the ordered union of the
// ranges defined by the provided KeyReaderSpecs. The union is computed with a
// bounded k-way merge: only the current head of each sub-reader is buffered,
// keys are never fully materialized in memory just to be sorted.
//
// All sub-readers are created through the transaction itself, therefore they
// share the transaction RYOW view and, per index, the same snapshot. Ranges
// backed by different index snapshots cannot be merged because the index key
// spaces are independent and the snapshots are not guaranteed to represent the
// same physical instant. All specs must also agree on the iteration direction
// (DescOrder), history mode (IncludeHistory), filter protocol (Filters) and
// offset, otherwise ErrIllegalArguments is returned.
//
// In normal mode each key is emitted at most once: overlapping ranges are
// deduplicated by raw key bytes and the observed copies must reference the
// same transaction version and value hash, otherwise the read fails. In
// IncludeHistory mode the identity of an entry is the (key, tx) pair, so all
// versions of a key are preserved while exact duplicates are removed; RYOW
// versions (tx 0) are ordered as the newest version of their key, matching the
// underlying history reader ordering.
//
// Filters and the unified offset are applied only after merging and
// deduplication. Stopping after a limited number of reads does not register
// any phantom check for ranges that were not fully drained. Every entry
// probed, filtered or prefetched from a sub-reader keeps its original MVCC
// read-set entry, so deduplication can never hide an underlying conflict.
//
// The returned key is owned by the caller and is not reused by subsequent
// Read calls.
func (tx *OngoingTx) NewMergedKeyReader(specs ...KeyReaderSpec) (KeyReader, error) {
	if tx.closed {
		return nil, ErrAlreadyClosed
	}

	if tx.IsWriteOnly() {
		return nil, ErrWriteOnlyTx
	}

	if len(specs) == 0 {
		return nil, fmt.Errorf("%w: at least one KeyReaderSpec is required", ErrIllegalArguments)
	}

	if err := validateMergedSpecs(specs); err != nil {
		return nil, err
	}

	snaps, err := tx.resolveMergedSnapshots(specs)
	if err != nil {
		return nil, err
	}

	subReaders, err := tx.newMergedSubReaders(specs, snaps)
	if err != nil {
		return nil, err
	}

	return &mergedKeyReader{
		tx:             tx,
		readers:        subReaders,
		filters:        specs[0].Filters,
		offset:         specs[0].Offset,
		descOrder:      specs[0].DescOrder,
		includeHistory: specs[0].IncludeHistory,
		heads:          make([]mergedReaderHead, len(subReaders)),
	}, nil
}

// validateMergedSpecs enforces the common direction, history mode, filter
// protocol and offset every merged range must follow.
func validateMergedSpecs(specs []KeyReaderSpec) error {
	for _, filter := range specs[0].Filters {
		if filter == nil {
			return fmt.Errorf("%w: invalid filter function", ErrIllegalArguments)
		}
	}

	for i := 1; i < len(specs); i++ {
		if specs[i].DescOrder != specs[0].DescOrder {
			return fmt.Errorf("%w: merged key reader requires a consistent DescOrder", ErrIllegalArguments)
		}

		if specs[i].IncludeHistory != specs[0].IncludeHistory {
			return fmt.Errorf("%w: merged key reader requires a consistent IncludeHistory", ErrIllegalArguments)
		}

		if specs[i].Offset != specs[0].Offset {
			return fmt.Errorf("%w: merged key reader requires a consistent Offset", ErrIllegalArguments)
		}

		if err := validateMergedFilters(specs[0].Filters, specs[i].Filters); err != nil {
			return err
		}
	}

	return nil
}

func validateMergedFilters(expected, actual []FilterFn) error {
	if len(actual) != len(expected) {
		return fmt.Errorf("%w: merged key reader requires a consistent filter protocol", ErrIllegalArguments)
	}

	for i := range expected {
		if reflect.ValueOf(actual[i]).Pointer() != reflect.ValueOf(expected[i]).Pointer() {
			return fmt.Errorf("%w: merged key reader requires a consistent filter protocol", ErrIllegalArguments)
		}
	}

	return nil
}

// resolveMergedSnapshots resolves every range through the transaction so that
// the RYOW interceptor is applied and per-index snapshots are reused, then
// rejects ranges backed by different index snapshots.
func (tx *OngoingTx) resolveMergedSnapshots(specs []KeyReaderSpec) ([]*Snapshot, error) {
	snaps := make([]*Snapshot, len(specs))

	for i, spec := range specs {
		snap, err := tx.snap(spec.Prefix)
		if err != nil {
			return nil, err
		}

		snaps[i] = snap
	}

	// Independent index snapshots do not share a physical time basis and key
	// spaces are disjoint trees, so raw-byte merging across indexes is not
	// allowed.
	for i := 1; i < len(snaps); i++ {
		if !bytes.Equal(snaps[i].prefix, snaps[0].prefix) {
			return nil, fmt.Errorf("%w: merged key reader ranges must belong to the same index", ErrIllegalArguments)
		}
	}

	return snaps, nil
}

// newMergedSubReaders creates one underlying key reader per range. Filters and
// offset are applied at merge level so that sub-readers keep recording every
// probed entry in the transaction MVCC read-set.
func (tx *OngoingTx) newMergedSubReaders(specs []KeyReaderSpec, snaps []*Snapshot) ([]KeyReader, error) {
	subReaders := make([]KeyReader, 0, len(specs))

	release := func() {
		for _, reader := range subReaders {
			_ = reader.Close()
		}
	}

	for i, spec := range specs {
		rspec := spec
		rspec.Filters = nil
		rspec.Offset = 0

		var reader KeyReader
		var err error

		if tx.IsReadOnly() {
			reader, err = snaps[i].NewKeyReader(rspec)
		} else {
			reader, err = newOngoingTxKeyReader(tx, rspec)
		}

		if err != nil {
			// On construction failure every sub-reader already created is
			// released. Snapshots and read-set entries registered up to the
			// failure stay owned by the transaction: already incurred
			// dependencies must not be lost.
			release()
			return nil, err
		}

		subReaders = append(subReaders, reader)
	}

	return subReaders, nil
}

type mergedReaderHead struct {
	loaded    bool
	exhausted bool
	key       []byte
	valRef    ValueRef
}

type mergedKeyReader struct {
	tx      *OngoingTx
	readers []KeyReader

	filters        []FilterFn
	offset         uint64
	skipped        uint64
	descOrder      bool
	includeHistory bool

	heads  []mergedReaderHead
	closed bool
}

func (r *mergedKeyReader) Read(ctx context.Context) (key []byte, valRef ValueRef, err error) {
	return r.readNext(ctx, 0, 0)
}

func (r *mergedKeyReader) ReadBetween(ctx context.Context, initialTxID, finalTxID uint64) (key []byte, valRef ValueRef, err error) {
	return r.readNext(ctx, initialTxID, finalTxID)
}

func (r *mergedKeyReader) readNext(ctx context.Context, initialTxID, finalTxID uint64) (key []byte, valRef ValueRef, err error) {
	if r.closed {
		return nil, nil, ErrAlreadyClosed
	}

	// ReadBetween always resolves the latest visible version per key, history
	// expansion only happens through plain Read calls.
	history := r.includeHistory && initialTxID == 0 && finalTxID == 0

	for {
		winner, pickErr := r.pickWinner(ctx, initialTxID, finalTxID, history)
		if pickErr != nil {
			return nil, nil, pickErr
		}

		if winner == -1 {
			return nil, nil, ErrNoMoreEntries
		}

		winHead := &r.heads[winner]
		winKey := winHead.key
		winTx := winHead.valRef.Tx()
		winHVal := winHead.valRef.HVal()
		outVal := winHead.valRef

		// The winning head is consumed; it gets refilled on the next read.
		winHead.loaded = false

		if dupErr := r.consumeDuplicates(winKey, winTx, winHVal, history); dupErr != nil {
			return nil, nil, dupErr
		}

		if r.isFiltered(outVal) {
			continue
		}

		// Offset applies to the merged, deduplicated stream only.
		if r.skipped < r.offset {
			r.skipped++
			continue
		}

		// Defensive copy: the caller owns the returned key and must not be
		// exposed to memory reused by the next Read call.
		return cp(winKey), outVal, nil
	}
}

// pickWinner fills the head of every non-exhausted sub-reader (at most one
// buffered entry per range) and returns the index of the head that must be
// emitted next. It returns -1 when every sub-reader reached its range end.
func (r *mergedKeyReader) pickWinner(ctx context.Context, initialTxID, finalTxID uint64, history bool) (int, error) {
	winner := -1

	for i := range r.readers {
		head := &r.heads[i]

		if !head.loaded && !head.exhausted {
			if err := r.fillHead(ctx, i, initialTxID, finalTxID); err != nil {
				// Budget exhaustion, cancellation or any other failure leaves
				// the merge state untouched: the caller can Close the reader
				// and the transaction dependencies remain.
				return 0, err
			}
		}

		if head.exhausted {
			continue
		}

		if winner == -1 || r.headBefore(i, winner, history) {
			winner = i
		}
	}

	return winner, nil
}

func (r *mergedKeyReader) fillHead(ctx context.Context, i, initialTxID, finalTxID uint64) error {
	var key []byte
	var valRef ValueRef
	var err error

	if initialTxID == 0 && finalTxID == 0 {
		key, valRef, err = r.readers[i].Read(ctx)
	} else {
		key, valRef, err = r.readers[i].ReadBetween(ctx, initialTxID, finalTxID)
	}

	if errors.Is(err, ErrNoMoreEntries) {
		// Reaching the end of a sub-reader registers the phantom check on
		// that range through its own read-set.
		r.heads[i].exhausted = true
		return nil
	}

	if err != nil {
		return err
	}

	r.heads[i].key = key
	r.heads[i].valRef = valRef
	r.heads[i].loaded = true

	return nil
}

// consumeDuplicates removes the winning entry from the other streams it was
// prefetched from. The prefetched copies are already part of the
// corresponding sub-reader MVCC read-set, so conflict detection is not
// weakened.
func (r *mergedKeyReader) consumeDuplicates(key []byte, tx uint64, hVal [sha256.Size]byte, history bool) error {
	for i := range r.readers {
		head := &r.heads[i]

		if !head.loaded || !r.sameEntry(head, key, tx, history) {
			continue
		}

		if head.valRef.HVal() != hVal || (!history && head.valRef.Tx() != tx) {
			return fmt.Errorf("%w: key %x", ErrMergedKeyReaderInconsistentEntries, key)
		}

		head.loaded = false
	}

	return nil
}

func (r *mergedKeyReader) isFiltered(valRef ValueRef) bool {
	for _, filter := range r.filters {
		if filter(valRef, r.tx.Timestamp()) != nil {
			return true
		}
	}

	return false
}

// headBefore reports whether the head of sub-reader i must be emitted before
// the head of sub-reader j. Ties keep the lower sub-reader index first so the
// merge is deterministic.
func (r *mergedKeyReader) headBefore(i, j int, history bool) bool {
	cmp := bytes.Compare(r.heads[i].key, r.heads[j].key)

	if cmp != 0 {
		if r.descOrder {
			return cmp > 0
		}

		return cmp < 0
	}

	if !history {
		return false
	}

	return versionBefore(r.heads[i].valRef.Tx(), r.heads[j].valRef.Tx(), r.descOrder)
}

// versionBefore orders two versions of the same key. The RYOW version (tx 0)
// represents the newest state of the key within the transaction and therefore
// sorts first in descending order and last in ascending order, matching the
// ordering produced by the underlying history reader.
func versionBefore(txA, txB uint64, descOrder bool) bool {
	if txA == txB {
		return false
	}

	if descOrder {
		if txA == 0 {
			return true
		}

		if txB == 0 {
			return false
		}

		return txA > txB
	}

	if txB == 0 {
		return true
	}

	if txA == 0 {
		return false
	}

	return txA < txB
}

func (r *mergedKeyReader) sameEntry(head *mergedReaderHead, key []byte, tx uint64, history bool) bool {
	if !bytes.Equal(head.key, key) {
		return false
	}

	if !history {
		return true
	}

	return head.valRef.Tx() == tx
}

// Reset rewinds the merge state and every sub-reader. Each sub-reader appends
// a new read trajectory to the transaction MVCC read-set without discarding the
// previously registered ones.
func (r *mergedKeyReader) Reset() error {
	if r.closed {
		return ErrAlreadyClosed
	}

	for _, reader := range r.readers {
		if err := reader.Reset(); err != nil {
			return err
		}
	}

	r.heads = make([]mergedReaderHead, len(r.readers))
	r.skipped = 0

	return nil
}

// Close releases every sub-reader. Snapshots themselves remain owned by the
// transaction and are released on Commit or Cancel.
func (r *mergedKeyReader) Close() error {
	if r.closed {
		return ErrAlreadyClosed
	}

	r.closed = true

	var errs []error

	for _, reader := range r.readers {
		if err := reader.Close(); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}
