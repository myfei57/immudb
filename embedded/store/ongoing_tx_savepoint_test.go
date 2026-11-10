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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func openSavepointTestStore(t *testing.T) *ImmuStore {
	st, err := Open(t.TempDir(), DefaultOptions().WithSynced(false))
	require.NoError(t, err)
	t.Cleanup(func() { immustoreClose(t, st) })
	return st
}

func mustCommit(t *testing.T, st *ImmuStore, entries map[string]string) {
	t.Helper()

	tx, err := st.NewWriteOnlyTx(context.Background())
	require.NoError(t, err)

	for k, v := range entries {
		require.NoError(t, tx.Set([]byte(k), nil, []byte(v)))
	}

	_, err = tx.Commit(context.Background())
	require.NoError(t, err)
}

func requireStoreGet(t *testing.T, st *ImmuStore, key, expected string) {
	t.Helper()

	valRef, err := st.Get(context.Background(), []byte(key))
	require.NoError(t, err)

	val, err := valRef.Resolve()
	require.NoError(t, err)
	require.Equal(t, []byte(expected), val)
}

func requireStoreKeyMissing(t *testing.T, st *ImmuStore, key string) {
	t.Helper()

	_, err := st.Get(context.Background(), []byte(key))
	require.ErrorIs(t, err, ErrKeyNotFound)
}

func requireTxGet(t *testing.T, tx *OngoingTx, key, expected string) {
	t.Helper()

	valRef, err := tx.Get(context.Background(), []byte(key))
	require.NoError(t, err)

	val, err := valRef.Resolve()
	require.NoError(t, err)
	require.Equal(t, []byte(expected), val)
}

func requireTxKeyMissing(t *testing.T, tx *OngoingTx, key string) {
	t.Helper()

	_, err := tx.Get(context.Background(), []byte(key))
	require.ErrorIs(t, err, ErrKeyNotFound)
}

// requireTxDeleted asserts the in-transaction view of a deleted key: the
// ongoing tx keeps exposing the tombstoned entry (with the Deleted metadata
// flag), mirroring the pre-existing Delete semantics inside an open tx.
func requireTxDeleted(t *testing.T, tx *OngoingTx, key string) {
	t.Helper()

	valRef, err := tx.Get(context.Background(), []byte(key))
	require.NoError(t, err)
	require.NotNil(t, valRef.KVMetadata())
	require.True(t, valRef.KVMetadata().Deleted())
}

func TestOngoingTxSavepointRejectsInvalidTokens(t *testing.T) {
	st := openSavepointTestStore(t)

	ctx := context.Background()

	t.Run("read only transactions cannot hold savepoints", func(t *testing.T) {
		roTx, err := st.NewTx(ctx, &TxOptions{Mode: ReadOnlyTx})
		require.NoError(t, err)
		defer roTx.Cancel()

		_, err = roTx.Savepoint()
		require.ErrorIs(t, err, ErrReadOnlyTx)

		err = roTx.RollbackToSavepoint(SavepointToken{})
		require.ErrorIs(t, err, ErrReadOnlyTx)

		err = roTx.ReleaseSavepoint(SavepointToken{})
		require.ErrorIs(t, err, ErrReadOnlyTx)
	})

	t.Run("zero value and foreign transaction tokens are rejected", func(t *testing.T) {
		tx1, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)
		defer tx1.Cancel()

		tx2, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)
		defer tx2.Cancel()

		err = tx1.RollbackToSavepoint(SavepointToken{})
		require.ErrorIs(t, err, ErrInvalidSavepointToken)

		err = tx1.ReleaseSavepoint(SavepointToken{})
		require.ErrorIs(t, err, ErrInvalidSavepointToken)

		token1, err := tx1.Savepoint()
		require.NoError(t, err)

		// tokens are bound to the originating transaction
		err = tx2.RollbackToSavepoint(token1)
		require.ErrorIs(t, err, ErrInvalidSavepointToken)

		err = tx2.ReleaseSavepoint(token1)
		require.ErrorIs(t, err, ErrInvalidSavepointToken)

		// the token still works on its own transaction
		require.NoError(t, tx1.RollbackToSavepoint(token1))
	})

	t.Run("tokens are rejected after cancel or commit", func(t *testing.T) {
		tx, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)

		token, err := tx.Savepoint()
		require.NoError(t, err)

		require.NoError(t, tx.Cancel())

		err = tx.RollbackToSavepoint(token)
		require.ErrorIs(t, err, ErrAlreadyClosed)

		err = tx.ReleaseSavepoint(token)
		require.ErrorIs(t, err, ErrAlreadyClosed)

		committed, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)

		committedToken, err := committed.Savepoint()
		require.NoError(t, err)

		require.NoError(t, committed.Set([]byte("ended"), nil, []byte("tx")))
		_, err = committed.Commit(ctx)
		require.NoError(t, err)

		err = committed.RollbackToSavepoint(committedToken)
		require.ErrorIs(t, err, ErrAlreadyClosed)
	})

	t.Run("released or invalidated tokens never work again", func(t *testing.T) {
		tx, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)
		defer tx.Cancel()

		sp1, err := tx.Savepoint()
		require.NoError(t, err)

		require.NoError(t, tx.Set([]byte("k"), nil, []byte("v")))

		sp2, err := tx.Savepoint()
		require.NoError(t, err)

		require.NoError(t, tx.ReleaseSavepoint(sp1))

		// release invalidates the target and every later savepoint
		err = tx.RollbackToSavepoint(sp1)
		require.ErrorIs(t, err, ErrInvalidSavepointToken)

		err = tx.RollbackToSavepoint(sp2)
		require.ErrorIs(t, err, ErrInvalidSavepointToken)
	})
}

func TestOngoingTxSavepointRollbackSameKeyRepeatedWrites(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	require.NoError(t, tx.Set([]byte("key"), nil, []byte("v1")))

	sp1, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("key"), nil, []byte("v2")))
	require.NoError(t, tx.Set([]byte("key"), nil, []byte("v3")))

	// repeated overwrites roll back in original operation order
	require.NoError(t, tx.RollbackToSavepoint(sp1))
	requireTxGet(t, tx, "key", "v1")

	// the target savepoint remains usable and new writes are possible
	require.NoError(t, tx.Set([]byte("key"), nil, []byte("v4")))

	sp2, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("key"), nil, []byte("v5")))
	require.NoError(t, tx.RollbackToSavepoint(sp2))
	requireTxGet(t, tx, "key", "v4")

	// repeated rollback to the same token is allowed
	require.NoError(t, tx.RollbackToSavepoint(sp2))
	requireTxGet(t, tx, "key", "v4")

	hdr, err := tx.Commit(ctx)
	require.NoError(t, err)
	require.EqualValues(t, 1, hdr.NEntries)

	// only the final write set is committed, history has a single version
	requireStoreGet(t, st, "key", "v4")

	_, hCount, err := st.History([]byte("key"), 0, false, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, hCount)
}

func TestOngoingTxSavepointRollbackNewKeysDeletesAndOverlays(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	mustCommit(t, st, map[string]string{"existing": "base"})

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	// rolling back the first in-tx write over a committed key restores the
	// committed view rather than hiding the key altogether
	sp0, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("existing"), nil, []byte("changed")))
	requireTxGet(t, tx, "existing", "changed")

	require.NoError(t, tx.RollbackToSavepoint(sp0))
	requireTxGet(t, tx, "existing", "base")

	// a brand new key disappears (snapshot placeholder is hidden)
	require.NoError(t, tx.Set([]byte("gone"), nil, []byte("x")))
	requireTxGet(t, tx, "gone", "x")

	require.NoError(t, tx.RollbackToSavepoint(sp0))
	requireTxKeyMissing(t, tx, "gone")

	// add then delete, rolled back to the state right after the add
	require.NoError(t, tx.Set([]byte("new3"), nil, []byte("z")))

	sp3, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Delete(ctx, []byte("new3")))
	requireTxDeleted(t, tx, "new3")

	require.NoError(t, tx.RollbackToSavepoint(sp3))
	requireTxGet(t, tx, "new3", "z")

	// deleting a committed key and rolling back brings the old value back
	spDel, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Delete(ctx, []byte("existing")))
	requireTxDeleted(t, tx, "existing")

	require.NoError(t, tx.RollbackToSavepoint(spDel))
	requireTxGet(t, tx, "existing", "base")

	// a key reader must not surface the rolled-back new key
	kr, err := tx.NewKeyReader(KeyReaderSpec{Prefix: []byte("gone")})
	require.NoError(t, err)

	_, _, err = kr.Read(ctx)
	require.ErrorIs(t, err, ErrNoMoreEntries)
	require.NoError(t, kr.Close())

	hdr, err := tx.Commit(ctx)
	require.NoError(t, err)

	// only new3 survives: the rolled-back "gone" add and "existing" delete
	// contribute nothing to the committed write set
	require.EqualValues(t, 1, hdr.NEntries)
	requireStoreGet(t, st, "new3", "z")
	requireStoreGet(t, st, "existing", "base")
	requireStoreKeyMissing(t, st, "gone")

	_, hCount, err := st.History([]byte("existing"), 0, false, 10)
	require.NoError(t, err)
	require.EqualValues(t, 1, hCount, "no tombstone must be committed for the rolled-back delete")
}

func TestOngoingTxSavepointMetadataAndIndependentOwnership(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	future := time.Now().Add(1 * time.Hour).Truncate(time.Second)

	md1 := NewKVMetadata()
	require.NoError(t, md1.ExpiresAt(future))

	valBuf := []byte("value1")

	// an early savepoint makes every subsequent write enter the transaction
	// through an independent, internally owned copy; the rollback target is
	// established only after the first write so update rollback can be tested
	_, err = tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("mk"), md1, valBuf))

	sp, err := tx.Savepoint()
	require.NoError(t, err)

	// caller mutates the slices it passed in before overwriting the entry:
	// the savepoint must keep an independently owned copy of the old record
	copy(valBuf, "999999")
	require.NoError(t, md1.ExpiresAt(time.Now().Add(24*time.Hour)))

	md2 := NewKVMetadata()
	require.NoError(t, md2.ExpiresAt(time.Now().Add(2*time.Hour)))
	require.NoError(t, tx.Set([]byte("mk"), md2, []byte("value2")))

	require.NoError(t, tx.RollbackToSavepoint(sp))

	valRef, err := tx.Get(context.Background(), []byte("mk"))
	require.NoError(t, err)

	val, err := valRef.Resolve()
	require.NoError(t, err)
	require.Equal(t, []byte("value1"), val, "journaled value must be immune to caller slice mutation")

	kvmd := valRef.KVMetadata()
	require.NotNil(t, kvmd)
	require.True(t, kvmd.IsExpirable())

	expAt, err := kvmd.ExpirationTime()
	require.NoError(t, err)
	require.Equal(t, future.Unix(), expAt.Unix(), "journaled metadata must be immune to caller mutation")

	// reusing the caller buffer after rollback must not change the restored
	// record either, since the tx now owns an internal copy
	copy(valBuf, "000000")

	valRef, err = tx.Get(context.Background(), []byte("mk"))
	require.NoError(t, err)

	val, err = valRef.Resolve()
	require.NoError(t, err)
	require.Equal(t, []byte("value1"), val)

	_, err = tx.Commit(ctx)
	require.NoError(t, err)

	requireStoreGet(t, st, "mk", "value1")
}

func TestKVMetadataCopy(t *testing.T) {
	require.Nil(t, (*KVMetadata)(nil).Copy())

	md := NewKVMetadata()
	require.NoError(t, md.AsDeleted(true))
	require.NoError(t, md.AsNonIndexable(true))

	expiresAt := time.Now().Add(time.Hour).Truncate(time.Second)
	require.NoError(t, md.ExpiresAt(expiresAt))

	c := md.Copy()
	require.NotSame(t, md, c)
	require.True(t, c.Deleted())
	require.True(t, c.NonIndexable())

	got, err := c.ExpirationTime()
	require.NoError(t, err)
	require.Equal(t, expiresAt.Unix(), got.Unix())

	// mutating the copy must not affect the original
	require.NoError(t, c.AsDeleted(false))
	require.NoError(t, c.AsNonIndexable(false))
	require.True(t, md.Deleted())
	require.True(t, md.NonIndexable())
}

func TestOngoingTxSavepointTransientEntries(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	sp, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.SetTransient([]byte("tk"), nil, []byte("tv")))
	requireTxGet(t, tx, "tk", "tv")

	require.NoError(t, tx.RollbackToSavepoint(sp))
	requireTxKeyMissing(t, tx, "tk")

	require.NoError(t, tx.SetTransient([]byte("tk2"), nil, []byte("tv2a")))

	sp2, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.SetTransient([]byte("tk2"), nil, []byte("tv2b")))
	requireTxGet(t, tx, "tk2", "tv2b")

	require.NoError(t, tx.RollbackToSavepoint(sp2))
	requireTxGet(t, tx, "tk2", "tv2a")

	// a durable entry is required to commit; transient entries never reach
	// the committed write set
	require.NoError(t, tx.Set([]byte("durable"), nil, []byte("yes")))

	_, err = tx.Commit(ctx)
	require.NoError(t, err)

	requireStoreGet(t, st, "durable", "yes")
	requireStoreKeyMissing(t, st, "tk")
	requireStoreKeyMissing(t, st, "tk2")
}

func TestOngoingTxSavepointNestedTokensAndRelease(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	sp1, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("a"), nil, []byte("1")))

	sp2, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("b"), nil, []byte("2")))

	sp3, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("c"), nil, []byte("3")))

	// rollback to sp2 restores the state at sp2 (only "a" exists) and
	// invalidates sp3: both "b" and "c" are undone
	require.NoError(t, tx.RollbackToSavepoint(sp2))

	requireTxGet(t, tx, "a", "1")
	requireTxKeyMissing(t, tx, "b")
	requireTxKeyMissing(t, tx, "c")

	err = tx.RollbackToSavepoint(sp3)
	require.ErrorIs(t, err, ErrInvalidSavepointToken)

	err = tx.ReleaseSavepoint(sp3)
	require.ErrorIs(t, err, ErrInvalidSavepointToken)

	// sp2 remains reusable, sp1 still works
	require.NoError(t, tx.RollbackToSavepoint(sp2))
	require.NoError(t, tx.RollbackToSavepoint(sp1))

	requireTxKeyMissing(t, tx, "a")
	requireTxKeyMissing(t, tx, "b")

	// release drops the target and every later savepoint, while earlier ones
	// keep working and the journal prefix is reclaimed
	require.NoError(t, tx.Set([]byte("d"), nil, []byte("4")))

	sp4, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("e"), nil, []byte("5")))

	sp5, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.ReleaseSavepoint(sp4))

	err = tx.RollbackToSavepoint(sp5)
	require.ErrorIs(t, err, ErrInvalidSavepointToken)

	// sp1 survives the release of later savepoints and still undoes writes
	require.NoError(t, tx.RollbackToSavepoint(sp1))
	requireTxKeyMissing(t, tx, "d")
	requireTxKeyMissing(t, tx, "e")

	// releasing the last live savepoint drains the journal completely
	require.NoError(t, tx.ReleaseSavepoint(sp1))
	require.Equal(t, 0, tx.savepointJournalLen())

	// fresh savepoints and writes keep working after a full release
	sp6, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("f"), nil, []byte("6")))
	require.NoError(t, tx.RollbackToSavepoint(sp6))
	requireTxKeyMissing(t, tx, "f")
}

func TestOngoingTxSavepointPreconditions(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	mustCommit(t, st, map[string]string{"present": "v"})

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	require.NoError(t, tx.Set([]byte("entry"), nil, []byte("kept")))

	sp, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.AddPrecondition(&PreconditionKeyMustExist{Key: []byte("missing")}))
	require.Len(t, tx.preconditions, 1)

	require.NoError(t, tx.RollbackToSavepoint(sp))
	require.Empty(t, tx.preconditions)

	// the failing precondition is gone, commit must succeed
	_, err = tx.Commit(ctx)
	require.NoError(t, err)

	// explicit preconditions added before the savepoint survive the rollback
	tx2, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx2.Cancel()

	require.NoError(t, tx2.AddPrecondition(&PreconditionKeyMustExist{Key: []byte("present")}))
	require.NoError(t, tx2.Set([]byte("entry2"), nil, []byte("x")))

	sp2, err := tx2.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx2.AddPrecondition(&PreconditionKeyMustExist{Key: []byte("missing")}))
	require.Len(t, tx2.preconditions, 2)

	require.NoError(t, tx2.RollbackToSavepoint(sp2))
	require.Len(t, tx2.preconditions, 1)

	_, err = tx2.Commit(ctx)
	require.NoError(t, err)
}

func TestOngoingTxSavepointRetainsReadDependencies(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	t.Run("observed get dependency survives the rollback (no write skew)", func(t *testing.T) {
		tx, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)
		defer tx.Cancel()

		// observe absence of "guard": this read dependency must not be
		// forgotten when a later write is rolled back
		_, err = tx.Get(ctx, []byte("guard"))
		require.ErrorIs(t, err, ErrKeyNotFound)

		readSetSizeBefore := tx.mvccReadSet.readsetSize
		require.Greater(t, readSetSizeBefore, 0)

		sp, err := tx.Savepoint()
		require.NoError(t, err)

		require.NoError(t, tx.Set([]byte("guard"), nil, []byte("local")))

		// another transaction commits concurrently and creates the key
		mustCommit(t, st, map[string]string{"guard": "remote"})

		require.NoError(t, tx.RollbackToSavepoint(sp))

		// the read budget is not refunded by the rollback: the observed
		// get dependency is still registered even though the write is gone
		require.Equal(t, readSetSizeBefore, tx.mvccReadSet.readsetSize)
		require.Len(t, tx.mvccReadSet.expectedGets, 1)
		require.Equal(t, []byte("guard"), tx.mvccReadSet.expectedGets[0].key)
		require.EqualValues(t, 0, tx.mvccReadSet.expectedGets[0].expectedTx)

		requireTxKeyMissing(t, tx, "guard")

		// an unrelated write still commits through the same transaction
		require.NoError(t, tx.Set([]byte("other"), nil, []byte("v")))

		_, err = tx.Commit(ctx)
		require.ErrorIs(t, err, ErrTxReadConflict)
	})

	t.Run("prefix fingerprint dependency survives the rollback", func(t *testing.T) {
		tx, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)
		defer tx.Cancel()

		require.NoError(t, tx.MarkPrefixScanned(ctx, KeyReaderSpec{Prefix: []byte("pfx")}))

		fpBefore := len(tx.mvccReadSet.expectedPrefixFPs)
		require.Equal(t, 1, fpBefore)

		sp, err := tx.Savepoint()
		require.NoError(t, err)

		require.NoError(t, tx.Set([]byte("pfx/local"), nil, []byte("v")))

		mustCommit(t, st, map[string]string{"pfx/remote": "v"})

		require.NoError(t, tx.RollbackToSavepoint(sp))

		require.Len(t, tx.mvccReadSet.expectedPrefixFPs, 1,
			"rollback must not drop observed prefix fingerprints")

		require.NoError(t, tx.Set([]byte("unrelated"), nil, []byte("v")))

		_, err = tx.Commit(ctx)
		require.ErrorIs(t, err, ErrTxReadConflict)
	})
}

func TestOngoingTxSavepointCommitFinalWriteSet(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	tx, err := st.NewTx(ctx, DefaultTxOptions())
	require.NoError(t, err)
	defer tx.Cancel()

	require.NoError(t, tx.Set([]byte("a"), nil, []byte("a1")))
	require.NoError(t, tx.Set([]byte("b"), nil, []byte("b1")))
	require.NoError(t, tx.Set([]byte("c"), nil, []byte("c1")))

	sp, err := tx.Savepoint()
	require.NoError(t, err)

	require.NoError(t, tx.Set([]byte("d"), nil, []byte("d1")))
	require.NoError(t, tx.Delete(ctx, []byte("a")))
	require.NoError(t, tx.Set([]byte("b"), nil, []byte("b2")))

	require.NoError(t, tx.RollbackToSavepoint(sp))

	// pending state is exactly the one captured by the savepoint
	requireTxGet(t, tx, "a", "a1")
	requireTxGet(t, tx, "b", "b1")
	requireTxGet(t, tx, "c", "c1")
	requireTxKeyMissing(t, tx, "d")

	hdr, err := tx.Commit(ctx)
	require.NoError(t, err)
	require.EqualValues(t, 3, hdr.NEntries)

	// persisted header confirms the rolled-back mutations never reached storage
	persisted, err := st.ReadTxHeader(hdr.ID, false, false)
	require.NoError(t, err)
	require.EqualValues(t, 3, persisted.NEntries)

	requireStoreGet(t, st, "a", "a1")
	requireStoreGet(t, st, "b", "b1")
	requireStoreGet(t, st, "c", "c1")
	requireStoreKeyMissing(t, st, "d")
}

func TestOngoingTxSavepointIncrementalJournalCost(t *testing.T) {
	st := openSavepointTestStore(t)
	ctx := context.Background()

	t.Run("cost grows with changes, not transaction size or savepoint count", func(t *testing.T) {
		tx, err := st.NewTx(ctx, DefaultTxOptions())
		require.NoError(t, err)
		defer tx.Cancel()

		const baseline = 300

		for i := 0; i < baseline; i++ {
			key := []byte{byte(i / 256), byte(i % 256)}
			require.NoError(t, tx.Set(key, nil, []byte("baseline")))
		}

		// no savepoint live yet: mutations are not journaled at all
		require.Equal(t, 0, tx.savepointJournalLen())
		require.EqualValues(t, 0, tx.undoRecordCount)

		sp, err := tx.Savepoint()
		require.NoError(t, err)

		// many savepoints without mutations between them add no journal records
		for i := 0; i < 200; i++ {
			_, err = tx.Savepoint()
			require.NoError(t, err)
		}

		require.Equal(t, 0, tx.savepointJournalLen())
		require.Equal(t, 201, tx.savepointCount())

		// new keys: one append record plus one snapshot overlay record each
		const newKeys = 25

		for i := 0; i < newKeys; i++ {
			key := []byte{0xEE, byte(i)}
			require.NoError(t, tx.Set(key, nil, []byte("new")))
		}

		require.Equal(t, 2*newKeys, tx.savepointJournalLen())
		require.Less(t, tx.savepointJournalLen(), baseline,
			"journal size must track the change count, not the full tx size")

		// in-place updates add a single record each and deep-copy the old spec
		const updates = 10

		for i := 0; i < updates; i++ {
			key := []byte{byte(i / 256), byte(i % 256)}
			require.NoError(t, tx.Set(key, nil, []byte("updated")))
		}

		require.Equal(t, 2*newKeys+updates, tx.savepointJournalLen())
		require.Greater(t, tx.undoOwnedBytes, 0)

		require.NoError(t, tx.RollbackToSavepoint(sp))
		require.Equal(t, 0, tx.savepointJournalLen())
		require.Len(t, tx.entries, baseline)
	})

	t.Run("write only transactions journal only entry mutations", func(t *testing.T) {
		tx, err := st.NewWriteOnlyTx(ctx)
		require.NoError(t, err)
		defer tx.Cancel()

		sp, err := tx.Savepoint()
		require.NoError(t, err)

		for i := 0; i < 5; i++ {
			require.NoError(t, tx.Set([]byte{byte(i)}, nil, []byte("v")))
		}

		// no snapshot overlaps exist for write-only transactions
		require.Equal(t, 5, tx.savepointJournalLen())

		require.NoError(t, tx.RollbackToSavepoint(sp))
		require.Empty(t, tx.entries)
	})
}
