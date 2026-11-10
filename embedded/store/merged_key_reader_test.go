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
	"sort"
	"testing"

	"github.com/stretchr/testify/require"
)

type mergedEntry struct {
	key    []byte
	tx     uint64
	hVal   [sha256.Size]byte
	valRef ValueRef
}

func drainMerged(t *testing.T, reader KeyReader) []mergedEntry {
	return drainMergedBetween(t, reader, 0, 0, false)
}

func drainMergedBetween(t *testing.T, reader KeyReader, initialTxID, finalTxID uint64, between bool) []mergedEntry {
	t.Helper()

	entries := make([]mergedEntry, 0)

	for {
		var key []byte
		var valRef ValueRef
		var err error

		if between {
			key, valRef, err = reader.ReadBetween(context.Background(), initialTxID, finalTxID)
		} else {
			key, valRef, err = reader.Read(context.Background())
		}

		if errors.Is(err, ErrNoMoreEntries) {
			break
		}

		require.NoError(t, err)

		entries = append(entries, mergedEntry{
			key:    cp(key),
			tx:     valRef.Tx(),
			hVal:   valRef.HVal(),
			valRef: valRef,
		})
	}

	return entries
}

// referenceMergedRead performs the union independently: every range is read
// fully with an existing single-range key reader on the same transaction, the
// collected entries are then sorted and deduplicated in memory. It encodes
// the same ordering and deduplication rules the streaming merged reader must
// follow.
func referenceMergedRead(t *testing.T, otx *OngoingTx, specs []KeyReaderSpec, between bool, initialTxID, finalTxID uint64) []mergedEntry {
	t.Helper()

	collected := make([]mergedEntry, 0)

	for _, spec := range specs {
		rspec := spec
		// filters and offset are applied once, at merge level
		rspec.Filters = nil
		rspec.Offset = 0

		reader, err := otx.NewKeyReader(rspec)
		require.NoError(t, err)

		for {
			var key []byte
			var valRef ValueRef

			if between {
				key, valRef, err = reader.ReadBetween(context.Background(), initialTxID, finalTxID)
			} else {
				key, valRef, err = reader.Read(context.Background())
			}

			if errors.Is(err, ErrNoMoreEntries) {
				break
			}

			require.NoError(t, err)

			collected = append(collected, mergedEntry{
				key:    cp(key),
				tx:     valRef.Tx(),
				hVal:   valRef.HVal(),
				valRef: valRef,
			})
		}

		require.NoError(t, reader.Close())
	}

	history := specs[0].IncludeHistory && !between
	descOrder := specs[0].DescOrder

	type entryID struct {
		key string
		tx  uint64
	}

	seen := make(map[entryID]int)
	deduped := make([]mergedEntry, 0, len(collected))

	for _, entry := range collected {
		id := entryID{key: string(entry.key)}

		if history {
			id.tx = entry.tx
		}

		if idx, ok := seen[id]; ok {
			require.Equal(t, deduped[idx].hVal, entry.hVal, "reference: inconsistent content for the same key version")

			if !history {
				require.Equal(t, deduped[idx].tx, entry.tx, "reference: inconsistent tx for the same key")
			}

			continue
		}

		seen[id] = len(deduped)
		deduped = append(deduped, entry)
	}

	sort.Slice(deduped, func(i, j int) bool {
		cmp := bytes.Compare(deduped[i].key, deduped[j].key)

		if cmp != 0 {
			if descOrder {
				return cmp > 0
			}

			return cmp < 0
		}

		if !history {
			return false
		}

		return versionBefore(deduped[i].tx, deduped[j].tx, descOrder)
	})

	// the uniform filter protocol is applied only after deduplication
	filtered := make([]mergedEntry, 0, len(deduped))

	for _, entry := range deduped {
		filterEntry := false

		for _, filter := range specs[0].Filters {
			if filter(entry.valRef, otx.Timestamp()) != nil {
				filterEntry = true
				break
			}
		}

		if !filterEntry {
			filtered = append(filtered, entry)
		}
	}

	// the unified offset is applied only after deduplication and filtering
	if specs[0].Offset >= uint64(len(filtered)) {
		return []mergedEntry{}
	}

	return filtered[specs[0].Offset:]
}

func assertSameSequence(t *testing.T, got, want []mergedEntry) {
	t.Helper()

	require.Len(t, got, len(want), "sequence length mismatch")

	for i := range want {
		require.Truef(t, bytes.Equal(got[i].key, want[i].key),
			"entry %d: key mismatch, got %x want %x", i, got[i].key, want[i].key)
		require.Equal(t, want[i].tx, got[i].tx, "entry %d (%x): tx mismatch", i, got[i].key)
		require.Equal(t, want[i].hVal, got[i].hVal, "entry %d (%x): content hash mismatch", i, got[i].key)
	}
}

func mergedTestStore(t *testing.T) *ImmuStore {
	st, err := Open(t.TempDir(), DefaultOptions().WithSynced(false).WithMaxConcurrency(4))
	require.NoError(t, err)
	return st
}

func commitWriteOnly(t *testing.T, st *ImmuStore, sets map[string][]byte, deletes ...string) {
	t.Helper()

	otx, err := st.NewWriteOnlyTx(context.Background())
	require.NoError(t, err)

	for key, val := range sets {
		require.NoError(t, otx.Set([]byte(key), nil, val))
	}

	for _, key := range deletes {
		md := NewKVMetadata()
		require.NoError(t, md.AsDeleted(true))
		require.NoError(t, otx.Set([]byte(key), md, nil))
	}

	_, err = otx.Commit(context.Background())
	require.NoError(t, err)
}

func TestMergedKeyReaderSequences(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	sets := make(map[string][]byte)

	for i := 0; i < 10; i++ {
		sets[formatMergedKey(i)] = []byte("v0")
	}

	commitWriteOnly(t, immuStore, sets)

	// a second transaction updates part of the keys so tx ids are not uniform
	commitWriteOnly(t, immuStore, map[string][]byte{
		formatMergedKey(3): []byte("v1"),
		formatMergedKey(7): []byte("v1"),
	})

	// committed deleted entry inside the overlapping union (observed by three
	// readers) to exercise the uniform filter protocol and deduplication
	commitWriteOnly(t, immuStore, map[string][]byte{}, formatMergedKey(6))

	baseSpecs := func() []KeyReaderSpec {
		return []KeyReaderSpec{
			{
				// key02..key07 inclusive
				Prefix:        []byte("key"),
				SeekKey:       []byte("key02"),
				EndKey:        []byte("key07"),
				InclusiveSeek: true,
				InclusiveEnd:  true,
			},
			{
				// key05..key08, end boundary excluded
				Prefix:        []byte("key"),
				SeekKey:       []byte("key05"),
				EndKey:        []byte("key09"),
				InclusiveSeek: true,
				InclusiveEnd:  false,
			},
			{
				// empty range: nothing starts at key20 under the prefix
				Prefix:        []byte("key"),
				SeekKey:       []byte("key20"),
				InclusiveSeek: true,
			},
			{
				// exclusive boundaries: only key03
				Prefix:        []byte("key"),
				SeekKey:       []byte("key02"),
				EndKey:        []byte("key04"),
				InclusiveSeek: false,
				InclusiveEnd:  false,
			},
			{
				// third overlapping range so that key06 is observed by 3 readers
				Prefix:        []byte("key"),
				SeekKey:       []byte("key06"),
				EndKey:        []byte("key06"),
				InclusiveSeek: true,
				InclusiveEnd:  true,
			},
		}
	}

	for _, descOrder := range []bool{false, true} {
		for _, offset := range []uint64{0, 2, 100} {
			for _, withFilter := range []bool{false, true} {
				otx, err := immuStore.NewTx(context.Background(),
					DefaultTxOptions().WithMode(ReadOnlyTx))
				require.NoError(t, err)

				specs := baseSpecs()

				for i := range specs {
					specs[i].DescOrder = descOrder
					specs[i].Offset = offset

					if withFilter {
						specs[i].Filters = []FilterFn{IgnoreDeleted}
					}
				}

				reader, err := otx.NewMergedKeyReader(specs...)
				require.NoError(t, err)

				got := drainMerged(t, reader)
				want := referenceMergedRead(t, otx, specs, false, 0, 0)

				assertSameSequence(t, got, want)

				_, _, err = reader.Read(context.Background())
				require.ErrorIs(t, err, ErrNoMoreEntries)

				require.NoError(t, reader.Close())
				require.NoError(t, otx.Cancel())
			}
		}
	}
}

func TestMergedKeyReaderIncludeHistory(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{
		"hA": []byte("a1"),
		"hB": []byte("b1"),
		"hC": []byte("c1"),
	})
	commitWriteOnly(t, immuStore, map[string][]byte{
		"hA": []byte("a2"),
		"hB": []byte("b2"),
	})
	commitWriteOnly(t, immuStore, map[string][]byte{
		"hA": []byte("a3"),
	})

	specs := func(descOrder bool) []KeyReaderSpec {
		return []KeyReaderSpec{
			{
				Prefix:         []byte("h"),
				SeekKey:        []byte("hA"),
				EndKey:         []byte("hB"),
				InclusiveSeek:  true,
				InclusiveEnd:   true,
				IncludeHistory: true,
				DescOrder:      descOrder,
			},
			{
				// overlaps on hB which has two historical versions
				Prefix:         []byte("h"),
				SeekKey:        []byte("hB"),
				EndKey:         []byte("hC"),
				InclusiveSeek:  true,
				InclusiveEnd:   true,
				IncludeHistory: true,
				DescOrder:      descOrder,
			},
		}
	}

	for _, descOrder := range []bool{false, true} {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		reader, err := otx.NewMergedKeyReader(specs(descOrder)...)
		require.NoError(t, err)

		got := drainMerged(t, reader)
		want := referenceMergedRead(t, otx, specs(descOrder), false, 0, 0)

		assertSameSequence(t, got, want)

		// hA has 3 versions, hB two, hC one, duplicates of hB removed
		require.Len(t, got, 3+2+1)

		require.NoError(t, reader.Close())

		// no concurrent change: history read-set must validate at commit time
		require.NoError(t, otx.Set("other", nil, []byte("write")))
		_, err = otx.Commit(context.Background())
		require.NoError(t, err)
	}

	// A RYOW-inserted key has no committed history: it yields a single
	// (key, tx 0) entry. In descending order that entry is the newest version
	// of its key and therefore sorts first.
	otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
	require.NoError(t, err)

	require.NoError(t, otx.Set("hD", nil, []byte("d-ryow")))

	ryowSpecs := []KeyReaderSpec{
		{
			Prefix:         []byte("h"),
			SeekKey:        []byte("hA"),
			EndKey:         []byte("hC"),
			InclusiveSeek:  true,
			InclusiveEnd:   true,
			IncludeHistory: true,
			DescOrder:      true,
		},
		{
			Prefix:         []byte("h"),
			SeekKey:        []byte("hC"),
			EndKey:         []byte("hD"),
			InclusiveSeek:  true,
			InclusiveEnd:   true,
			IncludeHistory: true,
			DescOrder:      true,
		},
	}

	reader, err := otx.NewMergedKeyReader(ryowSpecs...)
	require.NoError(t, err)

	got := drainMerged(t, reader)
	want := referenceMergedRead(t, otx, ryowSpecs, false, 0, 0)

	assertSameSequence(t, got, want)
	require.Equal(t, []byte("hD"), got[0].key)
	require.Equal(t, uint64(0), got[0].tx)

	ryowCount := 0

	for _, entry := range got {
		if bytes.Equal(entry.key, []byte("hD")) {
			ryowCount++
		}
	}

	require.Equal(t, 1, ryowCount)

	require.NoError(t, reader.Close())

	_, err = otx.Commit(context.Background())
	require.NoError(t, err)

	// a concurrent modification of a read key must still conflict
	otx2, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
	require.NoError(t, err)

	reader2, err := otx2.NewMergedKeyReader(specs(false)...)
	require.NoError(t, err)

	_ = drainMerged(t, reader2)
	require.NoError(t, reader2.Close())

	commitWriteOnly(t, immuStore, map[string][]byte{"hB": []byte("b3-concurrent")})

	require.NoError(t, otx2.Set("other2", nil, []byte("write")))
	_, err = otx2.Commit(context.Background())
	require.ErrorIs(t, err, ErrTxReadConflict)
}

func TestMergedKeyReaderRYOW(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{
		"r1": []byte("r1-old"),
		"r2": []byte("r2-old"),
		"r3": []byte("r3-old"),
	})

	otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
	require.NoError(t, err)

	require.NoError(t, otx.Set("r_new", nil, []byte("inserted")))
	require.NoError(t, otx.Set("r1", nil, []byte("r1-new")))
	require.NoError(t, otx.Delete(context.Background(), "r2"))

	specs := []KeyReaderSpec{
		{
			Prefix:        []byte("r"),
			SeekKey:       []byte("r1"),
			EndKey:        []byte("r2"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
		{
			Prefix:        []byte("r"),
			SeekKey:       []byte("r2"),
			EndKey:        []byte("r_new"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
	}

	// without filters the deleted entry is still visible
	reader, err := otx.NewMergedKeyReader(specs...)
	require.NoError(t, err)

	got := drainMerged(t, reader)
	want := referenceMergedRead(t, otx, specs, false, 0, 0)

	assertSameSequence(t, got, want)
	require.Len(t, got, 4)

	var r1ref ValueRef

	for _, entry := range got {
		if bytes.Equal(entry.key, []byte("r1")) {
			r1ref = entry.valRef
		}

		if bytes.Equal(entry.key, []byte("r2")) {
			require.NotNil(t, entry.valRef.KVMetadata())
			require.True(t, entry.valRef.KVMetadata().Deleted())
		}
	}

	val, err := r1ref.Resolve()
	require.NoError(t, err)
	require.Equal(t, []byte("r1-new"), val)

	require.NoError(t, reader.Close())

	// with IgnoreDeleted the RYOW-deleted key is filtered at merge level
	for i := range specs {
		specs[i].Filters = []FilterFn{IgnoreDeleted}
	}

	filteredReader, err := otx.NewMergedKeyReader(specs...)
	require.NoError(t, err)

	filtered := drainMerged(t, filteredReader)
	filteredWant := referenceMergedRead(t, otx, specs, false, 0, 0)

	assertSameSequence(t, filtered, filteredWant)
	require.Len(t, filtered, 3)

	for _, entry := range filtered {
		require.NotEqual(t, []byte("r2"), entry.key)
	}

	require.NoError(t, filteredReader.Close())

	_, err = otx.Commit(context.Background())
	require.NoError(t, err)
}

func TestMergedKeyReaderReadBetween(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{"b1": []byte("1"), "b2": []byte("1"), "b3": []byte("1")})
	commitWriteOnly(t, immuStore, map[string][]byte{"b1": []byte("2"), "b2": []byte("2")})
	commitWriteOnly(t, immuStore, map[string][]byte{"b1": []byte("3")})

	otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions().WithMode(ReadOnlyTx))
	require.NoError(t, err)

	specs := []KeyReaderSpec{
		{
			Prefix:        []byte("b"),
			SeekKey:       []byte("b1"),
			EndKey:        []byte("b2"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
		{
			Prefix:        []byte("b"),
			SeekKey:       []byte("b2"),
			EndKey:        []byte("b3"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
	}

	for _, window := range [][2]uint64{{1, 1}, {1, 2}, {2, 2}, {0, 2}} {
		reader, err := otx.NewMergedKeyReader(specs...)
		require.NoError(t, err)

		got := drainMergedBetween(t, reader, window[0], window[1], true)
		want := referenceMergedRead(t, otx, specs, true, window[0], window[1])

		assertSameSequence(t, got, want)

		require.NoError(t, reader.Close())
	}

	require.NoError(t, otx.Cancel())
}

func TestMergedKeyReaderConflicts(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{
		"c1": []byte("1"),
		"c2": []byte("1"),
		"c3": []byte("1"),
		"c5": []byte("1"),
		"c6": []byte("1"),
		"c7": []byte("1"),
	})

	mergedSpecs := []KeyReaderSpec{
		{
			Prefix:        []byte("c"),
			SeekKey:       []byte("c1"),
			EndKey:        []byte("c3"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
		{
			Prefix:        []byte("c"),
			SeekKey:       []byte("c5"),
			EndKey:        []byte("c7"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
	}

	// single-range reader covering the whole union, used to verify the merged
	// reader detects every conflict the original single-range reader detects
	coveringSpec := KeyReaderSpec{Prefix: []byte("c")}

	runScenario := func(t *testing.T, useMerged bool, drainFully bool, disturb func()) error {
		readerTx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		var reader KeyReader

		if useMerged {
			reader, err = readerTx.NewMergedKeyReader(mergedSpecs...)
		} else {
			spec := coveringSpec
			reader, err = readerTx.NewKeyReader(spec)
		}

		require.NoError(t, err)

		if drainFully {
			_ = drainMerged(t, reader)
		} else {
			_, _, err = reader.Read(context.Background())
			require.NoError(t, err)
			_, _, err = reader.Read(context.Background())
			require.NoError(t, err)
		}

		require.NoError(t, reader.Close())

		disturb()

		require.NoError(t, readerTx.Set("c_write", nil, []byte("v")))

		_, err = readerTx.Commit(context.Background())
		return err
	}

	disturbCommit := func(sets map[string][]byte, deletes ...string) func() {
		return func() {
			commitWriteOnly(t, immuStore, sets, deletes...)
		}
	}

	t.Run("phantom insert in fully drained overlapping ranges is detected", func(t *testing.T) {
		for _, useMerged := range []bool{true, false} {
			err := runScenario(t, useMerged, true, disturbCommit(map[string][]byte{"c55": []byte("phantom")}))
			require.ErrorIs(t, err, ErrTxReadConflict, "useMerged=%v", useMerged)
		}
	})

	t.Run("update of an overlapping key is detected", func(t *testing.T) {
		for _, useMerged := range []bool{true, false} {
			err := runScenario(t, useMerged, true, disturbCommit(map[string][]byte{"c6": []byte("updated")}))
			require.ErrorIs(t, err, ErrTxReadConflict, "useMerged=%v", useMerged)
		}
	})

	t.Run("delete of a read key is detected", func(t *testing.T) {
		for _, useMerged := range []bool{true, false} {
			err := runScenario(t, useMerged, true, disturbCommit(map[string][]byte{}, "c2"))
			require.ErrorIs(t, err, ErrTxReadConflict, "useMerged=%v", useMerged)
		}
	})

	t.Run("no conflict when the read key is updated by the ongoing transaction", func(t *testing.T) {
		for _, useMerged := range []bool{true, false} {
			readerTx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
			require.NoError(t, err)

			var reader KeyReader

			if useMerged {
				reader, err = readerTx.NewMergedKeyReader(mergedSpecs...)
			} else {
				reader, err = readerTx.NewKeyReader(coveringSpec)
			}

			require.NoError(t, err)
			_ = drainMerged(t, reader)
			require.NoError(t, reader.Close())

			require.NoError(t, readerTx.Set("c5", nil, []byte("self-update")))

			_, err = readerTx.Commit(context.Background())
			require.NoError(t, err, "useMerged=%v", useMerged)
		}
	})

	t.Run("partial reads do not register phantom checks for untouched ranges", func(t *testing.T) {
		// only the first heads of the sub-readers have been observed, so an
		// insertion beyond them must not invalidate the transaction
		err := runScenario(t, true, false, disturbCommit(map[string][]byte{"c9": []byte("far-away")}))
		require.NoError(t, err)
	})
}

func TestMergedKeyReaderReset(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{
		"d1": []byte("1"),
		"d2": []byte("1"),
		"d3": []byte("1"),
		"d4": []byte("1"),
	})

	specs := []KeyReaderSpec{
		{
			Prefix:        []byte("d"),
			SeekKey:       []byte("d1"),
			EndKey:        []byte("d2"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
		{
			Prefix:        []byte("d"),
			SeekKey:       []byte("d3"),
			EndKey:        []byte("d4"),
			InclusiveSeek: true,
			InclusiveEnd:  true,
		},
	}

	readerTx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
	require.NoError(t, err)

	reader, err := readerTx.NewMergedKeyReader(specs...)
	require.NoError(t, err)

	_, _, err = reader.Read(context.Background())
	require.NoError(t, err)

	require.NoError(t, reader.Reset())

	firstRun := drainMerged(t, reader)
	want := referenceMergedRead(t, readerTx, specs, false, 0, 0)
	assertSameSequence(t, firstRun, want)

	require.NoError(t, reader.Reset())

	secondRun := drainMerged(t, reader)
	assertSameSequence(t, secondRun, want)

	require.NoError(t, reader.Close())

	// both read trajectories remain registered: a concurrent update after the
	// reset is still detected at commit time
	commitWriteOnly(t, immuStore, map[string][]byte{"d3": []byte("concurrent")})

	_, err = readerTx.Commit(context.Background())
	require.ErrorIs(t, err, ErrTxReadConflict)
}

func TestMergedKeyReaderValidation(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{"m1": []byte("1"), "m2": []byte("1")})

	t.Run("no specs", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)
		defer otx.Cancel()

		_, err = otx.NewMergedKeyReader()
		require.ErrorIs(t, err, ErrIllegalArguments)
	})

	t.Run("inconsistent direction", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)
		defer otx.Cancel()

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m"), DescOrder: false},
			KeyReaderSpec{Prefix: []byte("m"), DescOrder: true},
		)
		require.ErrorIs(t, err, ErrIllegalArguments)
	})

	t.Run("inconsistent history mode", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)
		defer otx.Cancel()

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m"), IncludeHistory: true},
			KeyReaderSpec{Prefix: []byte("m"), IncludeHistory: false},
		)
		require.ErrorIs(t, err, ErrIllegalArguments)
	})

	t.Run("inconsistent offset", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)
		defer otx.Cancel()

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m"), Offset: 1},
			KeyReaderSpec{Prefix: []byte("m"), Offset: 2},
		)
		require.ErrorIs(t, err, ErrIllegalArguments)
	})

	t.Run("inconsistent filter protocol", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)
		defer otx.Cancel()

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m"), Filters: []FilterFn{IgnoreDeleted}},
			KeyReaderSpec{Prefix: []byte("m"), Filters: []FilterFn{IgnoreExpired}},
		)
		require.ErrorIs(t, err, ErrIllegalArguments)
	})

	t.Run("nil filter", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)
		defer otx.Cancel()

		_, err = otx.NewMergedKeyReader(KeyReaderSpec{
			Prefix:  []byte("m"),
			Filters: []FilterFn{nil},
		})
		require.ErrorIs(t, err, ErrIllegalArguments)
	})

	t.Run("write only and closed transactions are rejected", func(t *testing.T) {
		woTx, err := immuStore.NewWriteOnlyTx(context.Background())
		require.NoError(t, err)

		_, err = woTx.NewMergedKeyReader(KeyReaderSpec{Prefix: []byte("m")})
		require.ErrorIs(t, err, ErrWriteOnlyTx)

		require.NoError(t, woTx.Cancel())

		roTx, err := immuStore.NewTx(context.Background(), DefaultTxOptions().WithMode(ReadOnlyTx))
		require.NoError(t, err)
		require.NoError(t, roTx.Cancel())

		_, err = roTx.NewMergedKeyReader(KeyReaderSpec{Prefix: []byte("m")})
		require.ErrorIs(t, err, ErrAlreadyClosed)
	})

	t.Run("ranges from different indexes cannot be merged", func(t *testing.T) {
		multiStore, err := Open(t.TempDir(), DefaultOptions().WithSynced(false).WithMultiIndexing(true))
		require.NoError(t, err)
		defer immustoreClose(t, multiStore)

		require.NoError(t, multiStore.InitIndexing(&IndexSpec{
			SourcePrefix: []byte("j"),
			TargetPrefix: []byte("j"),
		}))
		require.NoError(t, multiStore.InitIndexing(&IndexSpec{
			SourcePrefix: []byte("k"),
			TargetPrefix: []byte("k"),
		}))

		commitWriteOnly(t, multiStore, map[string][]byte{
			"j1": []byte("1"),
			"k1": []byte("1"),
			"m1": []byte("1"),
			"m4": []byte("1"),
		})

		otx, err := multiStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("j1")},
			KeyReaderSpec{Prefix: []byte("k1")},
		)
		require.ErrorIs(t, err, ErrIllegalArguments)

		// different sub-prefixes within the same default index are mergeable
		reader, err := otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m"), SeekKey: []byte("m1"), EndKey: []byte("m2"), InclusiveSeek: true, InclusiveEnd: true},
			KeyReaderSpec{Prefix: []byte("m"), SeekKey: []byte("m4"), EndKey: []byte("m5"), InclusiveSeek: true, InclusiveEnd: true},
		)
		require.NoError(t, err)

		entries := drainMerged(t, reader)
		require.Len(t, entries, 2)

		require.NoError(t, reader.Close())
		require.NoError(t, otx.Cancel())
	})

	t.Run("construction failure releases the already created sub-readers", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		invalidSeek := make([]byte, immuStore.maxKeyLen+1)

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m")},
			KeyReaderSpec{Prefix: []byte("m"), SeekKey: invalidSeek},
		)
		require.ErrorIs(t, err, ErrIllegalArguments)

		// no dangling reader must prevent the transaction cleanup
		require.NoError(t, otx.Cancel())
	})

	t.Run("double close is rejected", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions().WithMode(ReadOnlyTx))
		require.NoError(t, err)

		reader, err := otx.NewMergedKeyReader(KeyReaderSpec{Prefix: []byte("m")})
		require.NoError(t, err)

		require.NoError(t, reader.Close())
		require.ErrorIs(t, reader.Close(), ErrAlreadyClosed)
		require.NoError(t, otx.Cancel())
	})

	t.Run("open reader blocks commit", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		reader, err := otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("m")},
			KeyReaderSpec{Prefix: []byte("m"), SeekKey: []byte("m1"), EndKey: []byte("m1"), InclusiveSeek: true, InclusiveEnd: true},
		)
		require.NoError(t, err)

		_, err = otx.Commit(context.Background())
		require.Error(t, err)

		require.NoError(t, reader.Close())
		require.NoError(t, otx.Cancel())
	})
}

func TestMergedKeyReaderReadSetLimit(t *testing.T) {
	immuStore, err := Open(t.TempDir(),
		DefaultOptions().WithSynced(false).WithMVCCReadSetLimit(4))
	require.NoError(t, err)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{
		"l1": []byte("1"),
		"l2": []byte("1"),
		"l3": []byte("1"),
		"l4": []byte("1"),
		"l5": []byte("1"),
	})

	t.Run("budget exhaustion while reading is propagated and cleanup works", func(t *testing.T) {
		otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		reader, err := otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("l"), SeekKey: []byte("l1"), EndKey: []byte("l3"), InclusiveSeek: true, InclusiveEnd: true},
			KeyReaderSpec{Prefix: []byte("l"), SeekKey: []byte("l4"), EndKey: []byte("l5"), InclusiveSeek: true, InclusiveEnd: true},
		)
		require.NoError(t, err)

		limitErr := false

		for {
			_, _, rerr := reader.Read(context.Background())
			if errors.Is(rerr, ErrNoMoreEntries) {
				break
			}

			if errors.Is(rerr, ErrMVCCReadSetLimitExceeded) {
				limitErr = true
				break
			}

			require.NoError(t, rerr)
		}

		require.True(t, limitErr)
		require.NoError(t, reader.Close())
		require.NoError(t, otx.Cancel())
	})

	t.Run("budget exhaustion during construction closes created readers", func(t *testing.T) {
		limitedStore, lerr := Open(t.TempDir(),
			DefaultOptions().WithSynced(false).WithMVCCReadSetLimit(1))
		require.NoError(t, lerr)
		defer immustoreClose(t, limitedStore)

		commitWriteOnly(t, limitedStore, map[string][]byte{"l1": []byte("1"), "l2": []byte("1")})

		otx, err := limitedStore.NewTx(context.Background(), DefaultTxOptions())
		require.NoError(t, err)

		_, err = otx.NewMergedKeyReader(
			KeyReaderSpec{Prefix: []byte("l"), SeekKey: []byte("l1"), EndKey: []byte("l1"), InclusiveSeek: true, InclusiveEnd: true},
			KeyReaderSpec{Prefix: []byte("l"), SeekKey: []byte("l2"), EndKey: []byte("l2"), InclusiveSeek: true, InclusiveEnd: true},
		)
		require.ErrorIs(t, err, ErrMVCCReadSetLimitExceeded)

		require.NoError(t, otx.Cancel())
	})
}

func TestMergedKeyReaderMemoryOwnership(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{
		"o1": []byte("1"),
		"o2": []byte("2"),
		"o3": []byte("3"),
	})

	otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions().WithMode(ReadOnlyTx))
	require.NoError(t, err)

	reader, err := otx.NewMergedKeyReader(
		KeyReaderSpec{Prefix: []byte("o"), SeekKey: []byte("o1"), EndKey: []byte("o2"), InclusiveSeek: true, InclusiveEnd: true},
		KeyReaderSpec{Prefix: []byte("o"), SeekKey: []byte("o2"), EndKey: []byte("o3"), InclusiveSeek: true, InclusiveEnd: true},
	)
	require.NoError(t, err)

	key, _, err := reader.Read(context.Background())
	require.NoError(t, err)
	require.Equal(t, []byte("o1"), key)

	// mutating the returned key must not corrupt any internal buffer
	for i := range key {
		key[i] = 0xFF
	}

	key2, _, err := reader.Read(context.Background())
	require.NoError(t, err)
	require.Equal(t, []byte("o2"), key2)

	key3, _, err := reader.Read(context.Background())
	require.NoError(t, err)
	require.Equal(t, []byte("o3"), key3)

	require.NoError(t, reader.Close())
	require.NoError(t, otx.Cancel())
}

func TestMergedKeyReaderContextCancellation(t *testing.T) {
	immuStore := mergedTestStore(t)
	defer immustoreClose(t, immuStore)

	commitWriteOnly(t, immuStore, map[string][]byte{"p1": []byte("1"), "p2": []byte("2")})

	otx, err := immuStore.NewTx(context.Background(), DefaultTxOptions())
	require.NoError(t, err)

	reader, err := otx.NewMergedKeyReader(
		KeyReaderSpec{Prefix: []byte("p"), SeekKey: []byte("p1"), EndKey: []byte("p1"), InclusiveSeek: true, InclusiveEnd: true},
		KeyReaderSpec{Prefix: []byte("p"), SeekKey: []byte("p2"), EndKey: []byte("p2"), InclusiveSeek: true, InclusiveEnd: true},
	)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, err = reader.Read(ctx)
	require.ErrorIs(t, err, context.Canceled)

	require.NoError(t, reader.Close())
	require.NoError(t, otx.Cancel())
}

func formatMergedKey(i int) string {
	return "key" + string(rune('0'+i/10)) + string(rune('0'+i%10))
}
