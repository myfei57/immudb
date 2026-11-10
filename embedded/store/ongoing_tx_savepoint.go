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
	"crypto/sha256"
	"errors"
	"sync/atomic"
)

var (
	// ErrInvalidSavepointToken is returned when a savepoint token does not
	// reference a live savepoint of the transaction it is presented to.
	// This covers zero-value tokens, tokens minted by another transaction,
	// tokens whose savepoint has been released or invalidated by a rollback,
	// and tokens presented after the transaction has been committed or
	// cancelled.
	ErrInvalidSavepointToken = errors.New("invalid savepoint token")
)

// ongoingTxTokenCounter assigns each OngoingTx an unique, process-local
// identity used to bind savepoint tokens to their originating transaction.
var ongoingTxTokenCounter atomic.Uint64

// SavepointToken is an opaque, transaction-bound handle to a savepoint
// established with OngoingTx.Savepoint.
//
// A token is bound to:
//   - the transaction that minted it (txID): presenting it to any other
//     transaction, or to the same transaction after it ended (Commit,
//     AsyncCommit or Cancel), is rejected with ErrInvalidSavepointToken;
//   - a savepoint generation (gen): once the generation is invalidated by a
//     RollbackToSavepoint to an earlier savepoint or by ReleaseSavepoint,
//     the token can never be reused.
//
// Token values carry no meaningful state for callers and must not be
// constructed manually.
type SavepointToken struct {
	txID uint64
	gen  uint64
}

// savepointState is the internal, mutable counterpart of a SavepointToken.
// mark is the absolute undo-journal sequence number at creation time:
// rolling back to the savepoint undoes every journal record whose sequence
// is greater than or equal to mark.
type savepointState struct {
	gen  uint64
	mark uint64
}

// undoFn restores a single state mutation performed inside the ongoing
// transaction. Undo functions are applied in reverse (LIFO) order by
// RollbackToSavepoint.
type undoFn func()

// restoredOverlay keeps the indexed value a read snapshot held before an
// in-transaction placeholder overwrote it. tbtree snapshots cannot replace a
// value at the same timestamp, so rolling back the placeholder insertion
// restores the previous value through the snapshot interceptor instead of
// writing the tree again.
type restoredOverlay struct {
	indexedVal []byte
	ts         uint64
	hc         uint64
}

// ongoingHiddenValRef is the sentinel ValueRef returned by the transaction
// snapshot interceptor for keys whose in-transaction insertion was undone
// by a savepoint rollback. The underlying read-only snapshot still contains
// the placeholder entry inserted via Snapshot.set when the key was first
// written, so read paths detect this sentinel and behave as if the key were
// absent instead of leaking a placeholder value reference.
type ongoingHiddenValRef struct{}

var hiddenSnapshotValRef ValueRef = &ongoingHiddenValRef{}

func (*ongoingHiddenValRef) Resolve() ([]byte, error) { return nil, ErrKeyNotFound }
func (*ongoingHiddenValRef) Tx() uint64               { return 0 }
func (*ongoingHiddenValRef) HC() uint64               { return 0 }
func (*ongoingHiddenValRef) TxMetadata() *TxMetadata  { return nil }
func (*ongoingHiddenValRef) KVMetadata() *KVMetadata  { return nil }
func (*ongoingHiddenValRef) HVal() [sha256.Size]byte  { return [sha256.Size]byte{} }
func (*ongoingHiddenValRef) Len() uint32              { return 0 }
func (*ongoingHiddenValRef) VOff() int64              { return 0 }

func isHiddenSnapshotValRef(valRef ValueRef) bool {
	_, ok := valRef.(*ongoingHiddenValRef)
	return ok
}

// Savepoint establishes a savepoint covering the current pending state of
// the transaction: all entries, transient entries, deletions and explicit
// preconditions written so far. It returns an opaque token bound to this
// transaction and to a fresh savepoint generation.
//
// Establishing a savepoint does not copy the transaction state: mutations
// performed after the savepoint append incremental records to an undo
// journal, so the cost is independent of the transaction size.
//
// Read-only transactions cannot hold savepoints.
func (tx *OngoingTx) Savepoint() (SavepointToken, error) {
	if tx.closed {
		return SavepointToken{}, ErrAlreadyClosed
	}

	if tx.IsReadOnly() {
		return SavepointToken{}, ErrReadOnlyTx
	}

	if tx.savepoints == nil {
		tx.savepoints = make(map[uint64]*savepointState)
	}

	tx.nextSavepointGen++

	gen := tx.nextSavepointGen

	tx.savepoints[gen] = &savepointState{
		gen:  gen,
		mark: tx.journalSeq(),
	}

	return SavepointToken{txID: tx.txTokenID, gen: gen}, nil
}

// RollbackToSavepoint restores the pending transaction state captured when
// the savepoint identified by token was established: writes, deletions,
// transient entries and explicit precondition changes performed afterwards
// are undone in reverse operation order.
//
// Every savepoint established after the target savepoint is invalidated and
// its token will be rejected from then on. The target savepoint itself is
// retained and remains usable, including repeated rollbacks. The MVCC
// read-set (observed gets, readers and prefix fingerprints) and the read
// budget consumption are deliberately left untouched: a rollback must not
// forget read dependencies, otherwise write skew could be introduced.
//
// Rollback only mutates in-memory transaction state. It never creates a
// persistent transaction, writes the WAL or triggers indexing.
func (tx *OngoingTx) RollbackToSavepoint(token SavepointToken) error {
	sp, err := tx.resolveSavepoint(token)
	if err != nil {
		return err
	}

	// Apply undo records in reverse order until reaching the savepoint mark.
	for seq := tx.journalSeq(); seq > sp.mark; seq-- {
		idx := seq - tx.journalBase - 1
		fn := tx.undoJournal[idx]
		fn()
		tx.undoJournal[idx] = nil // release references captured by the closure
	}

	keep := int(sp.mark - tx.journalBase)
	tx.undoJournal = tx.undoJournal[:keep]

	// Invalidate every savepoint created after the target one. The target
	// generation is retained so the token stays reusable.
	for gen := range tx.savepoints {
		if gen > sp.gen {
			delete(tx.savepoints, gen)
		}
	}

	return nil
}

// ReleaseSavepoint disposes of the named savepoint and of every savepoint
// established after it: their tokens are rejected from then on. Undo
// journal records that no surviving savepoint can roll back to are
// discarded as well, bounding the journal size to the amount of still
// undoable change rather than to the whole transaction history.
//
// Releasing a savepoint does not change the pending transaction state.
func (tx *OngoingTx) ReleaseSavepoint(token SavepointToken) error {
	sp, err := tx.resolveSavepoint(token)
	if err != nil {
		return err
	}

	for gen := range tx.savepoints {
		if gen >= sp.gen {
			delete(tx.savepoints, gen)
		}
	}

	tx.discardUnreachableJournal()

	return nil
}

// resolveSavepoint validates that token references a live savepoint of this
// very transaction.
func (tx *OngoingTx) resolveSavepoint(token SavepointToken) (*savepointState, error) {
	if tx.closed {
		return nil, ErrAlreadyClosed
	}

	if tx.IsReadOnly() {
		return nil, ErrReadOnlyTx
	}

	// The txID check rejects zero-value tokens and tokens minted by another
	// (including an already ended) transaction.
	if token.txID != tx.txTokenID {
		return nil, ErrInvalidSavepointToken
	}

	sp, ok := tx.savepoints[token.gen]
	if !ok {
		return nil, ErrInvalidSavepointToken
	}

	return sp, nil
}

// discardUnreachableJournal drops journal prefixes that no surviving
// savepoint can possibly roll back to, adjusting the absolute sequence base.
func (tx *OngoingTx) discardUnreachableJournal() {
	if len(tx.savepoints) == 0 {
		tx.journalBase += uint64(len(tx.undoJournal))

		for i := range tx.undoJournal {
			tx.undoJournal[i] = nil
		}

		tx.undoJournal = tx.undoJournal[:0]
		return
	}

	var minMark uint64 = ^uint64(0)

	for _, sp := range tx.savepoints {
		if sp.mark < minMark {
			minMark = sp.mark
		}
	}

	if minMark <= tx.journalBase {
		return
	}

	drop := int(minMark - tx.journalBase)

	for i := 0; i < drop; i++ {
		tx.undoJournal[i] = nil
	}

	tx.undoJournal = tx.undoJournal[drop:]
	tx.journalBase = minMark
}

// releaseSavepointState frees every savepoint, undo journal record and
// hidden-key marker held by the transaction. It is invoked when the
// transaction ends through Commit, AsyncCommit or Cancel, after which all
// tokens are rejected anyway.
func (tx *OngoingTx) releaseSavepointState() {
	for i := range tx.undoJournal {
		tx.undoJournal[i] = nil
	}

	tx.undoJournal = nil
	tx.journalBase = 0
	tx.savepoints = nil
	tx.nextSavepointGen = 0
	tx.hiddenKeys = nil
	tx.overlayRestored = nil
}

// journalingActive reports whether at least one live savepoint exists.
// Mutations are only journaled while a savepoint can roll back to them, so
// transactions that never use savepoints pay no journaling cost.
func (tx *OngoingTx) journalingActive() bool {
	return len(tx.savepoints) > 0
}

// journalSeq returns the next absolute undo-journal sequence number.
func (tx *OngoingTx) journalSeq() uint64 {
	return tx.journalBase + uint64(len(tx.undoJournal))
}

func (tx *OngoingTx) appendUndo(fn undoFn) {
	tx.undoJournal = append(tx.undoJournal, fn)
	tx.undoRecordCount++
}

func (tx *OngoingTx) addHiddenKey(kid [sha256.Size]byte) {
	if tx.hiddenKeys == nil {
		tx.hiddenKeys = make(map[[sha256.Size]byte]struct{})
	}

	tx.hiddenKeys[kid] = struct{}{}
}

func (tx *OngoingTx) isKeyHidden(kid [sha256.Size]byte) bool {
	_, hidden := tx.hiddenKeys[kid]
	return hidden
}

// journalEntryAppend records the undo of a non-transient entry append:
// removing the key from entriesByKey and truncating tx.entries back.
// Appends are always undone in LIFO order, so all later appends have
// already been undone when this runs and idx is still the last slot.
func (tx *OngoingTx) journalEntryAppend(kid [sha256.Size]byte, idx int) {
	tx.appendUndo(func() {
		delete(tx.entriesByKey, kid)
		tx.entries = tx.entries[:idx]
	})
}

// journalEntryUpdate records the undo of an in-place replacement of a
// non-transient entry. The previous spec is deep copied so later caller
// mutations of the original key/value/metadata slices, or buffer reuse
// after a rollback, cannot alter the captured state.
func (tx *OngoingTx) journalEntryUpdate(idx int, old *EntrySpec) {
	saved := cloneEntrySpec(old)
	tx.undoOwnedBytes += len(saved.Key) + len(saved.Value)

	tx.appendUndo(func() {
		tx.entries[idx] = saved
	})
}

// journalTransientAdd records the undo of a transient entry insertion.
func (tx *OngoingTx) journalTransientAdd(kid [sha256.Size]byte, ref int) {
	tx.appendUndo(func() {
		delete(tx.entriesByKey, kid)
		delete(tx.transientEntries, ref)
	})
}

// journalTransientUpdate records the undo of an in-place replacement of a
// transient entry with an independently owned copy of the previous spec.
func (tx *OngoingTx) journalTransientUpdate(ref int, old *EntrySpec) {
	saved := cloneEntrySpec(old)
	tx.undoOwnedBytes += len(saved.Key) + len(saved.Value)

	tx.appendUndo(func() {
		tx.transientEntries[ref] = saved
	})
}

// journalOverlayMutation records the undo of a Snapshot.set placeholder
// about to be inserted while writing a key that was not yet part of the
// transaction write-set. It must be invoked BEFORE the placeholder is set
// on the snapshot so the previous indexed value is still observable.
//
// A tbtree snapshot cannot replace an entry written at its own timestamp,
// therefore the previous state is not restored by writing the snapshot
// again: if the key previously existed (for example a first write over a
// committed key), its previous indexed value is captured and handed back to
// readers through the snapshot interceptor; otherwise the key is marked
// hidden so read paths treat it as absent once the insertion is rolled back.
func (tx *OngoingTx) journalOverlayMutation(snap *Snapshot, targetKey []byte) error {
	if !tx.journalingActive() {
		return nil
	}

	kid := sha256.Sum256(targetKey)

	// A previous rollback may already have detached an overlay for this key.
	// Re-capture that restored state: the tree itself still holds the stale
	// same-timestamp placeholder and cannot be read for the old value.
	if restored, ok := tx.overlayRestored[kid]; ok {
		tx.appendUndo(func() {
			tx.putRestoredOverlay(kid, restored)
		})

		return nil
	}

	if tx.isKeyHidden(kid) {
		tx.appendUndo(func() {
			tx.addHiddenKey(kid)
		})

		return nil
	}

	prevVal, ts, hc, err := snap.snap.Get(targetKey)
	if errors.Is(err, ErrKeyNotFound) {
		tx.appendUndo(func() {
			tx.addHiddenKey(kid)
		})

		return nil
	}

	if err != nil {
		return err
	}

	saved := restoredOverlay{
		indexedVal: cp(prevVal),
		ts:         ts,
		hc:         hc,
	}

	tx.appendUndo(func() {
		tx.putRestoredOverlay(kid, saved)
	})

	return nil
}

func (tx *OngoingTx) putRestoredOverlay(kid [sha256.Size]byte, restored restoredOverlay) {
	if tx.overlayRestored == nil {
		tx.overlayRestored = make(map[[sha256.Size]byte]restoredOverlay)
	}

	tx.overlayRestored[kid] = restored
}

// cloneEntrySpec returns a deep, independently owned copy of an EntrySpec.
func cloneEntrySpec(e *EntrySpec) *EntrySpec {
	if e == nil {
		return nil
	}

	return &EntrySpec{
		Key:              cp(e.Key),
		Metadata:         e.Metadata.Copy(),
		Value:            cp(e.Value),
		HashValue:        e.HashValue,
		IsValueTruncated: e.IsValueTruncated,
	}
}

// savepointJournalLen exposes the current number of live undo records. It
// is intended for in-package cost accounting tests.
func (tx *OngoingTx) savepointJournalLen() int {
	return len(tx.undoJournal)
}

// savepointCount exposes the number of currently live savepoints. It is
// intended for in-package cost accounting tests.
func (tx *OngoingTx) savepointCount() int {
	return len(tx.savepoints)
}
