// Package benchmarks provides performance benchmarks for the Go core.
package benchmarks

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/VictoriaMetrics/fastcache"
)

// ════════════════════════════════════════════════════════════════════════════════
// FASTCACHE BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════════

func BenchmarkFastCacheGet(b *testing.B) {
	cache := fastcache.New(1 << 30) // 1GB
	
	// Pre-populate cache
	for i := 0; i < 10000; i++ {
		key := []byte(fmt.Sprintf("key-%d", i))
		value := make([]byte, 1024) // 1KB values
		rand.Read(value)
		cache.Set(key, value)
	}
	
	keys := make([][]byte, 10000)
	for i := 0; i < 10000; i++ {
		keys[i] = []byte(fmt.Sprintf("key-%d", i))
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			idx := rand.Intn(10000)
			cache.Get(nil, keys[idx])
		}
	})
}

func BenchmarkFastCacheSet(b *testing.B) {
	cache := fastcache.New(1 << 30) // 1GB
	value := make([]byte, 1024) // 1KB values
	rand.Read(value)
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := []byte(fmt.Sprintf("key-%d", i))
			cache.Set(key, value)
			i++
		}
	})
}

func BenchmarkFastCacheMixed(b *testing.B) {
	cache := fastcache.New(1 << 30) // 1GB
	
	// Pre-populate cache
	for i := 0; i < 10000; i++ {
		key := []byte(fmt.Sprintf("key-%d", i))
		value := make([]byte, 1024)
		rand.Read(value)
		cache.Set(key, value)
	}
	
	newValue := make([]byte, 1024)
	rand.Read(newValue)
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%10 == 0 { // 10% writes
				key := []byte(fmt.Sprintf("new-key-%d", i))
				cache.Set(key, newValue)
			} else { // 90% reads
				key := []byte(fmt.Sprintf("key-%d", rand.Intn(10000)))
				cache.Get(nil, key)
			}
			i++
		}
	})
}

// ════════════════════════════════════════════════════════════════════════════════
// CONTEXT BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════════

func BenchmarkContextWithTimeout(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = ctx
			cancel()
		}
	})
}

// ════════════════════════════════════════════════════════════════════════════════
// GOROUTINE BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════════

func BenchmarkGoroutineSpawn(b *testing.B) {
	done := make(chan struct{})
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		go func() {
			done <- struct{}{}
		}()
		<-done
	}
}

func BenchmarkChannelSend(b *testing.B) {
	ch := make(chan int, 1000)
	
	go func() {
		for range ch {
		}
	}()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ch <- i
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// MEMORY BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════════

func BenchmarkSliceAllocation(b *testing.B) {
	b.Run("1KB", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = make([]byte, 1024)
		}
	})
	
	b.Run("1MB", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = make([]byte, 1024*1024)
		}
	})
	
	b.Run("10MB", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = make([]byte, 10*1024*1024)
		}
	})
}

func BenchmarkSlicePreallocation(b *testing.B) {
	b.Run("NoPrealloc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			s := make([]int, 0)
			for j := 0; j < 1000; j++ {
				s = append(s, j)
			}
		}
	})
	
	b.Run("WithPrealloc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			s := make([]int, 0, 1000)
			for j := 0; j < 1000; j++ {
				s = append(s, j)
			}
		}
	})
}

// ════════════════════════════════════════════════════════════════════════════════
// MAP BENCHMARKS
// ════════════════════════════════════════════════════════════════════════════════

func BenchmarkMapAccess(b *testing.B) {
	m := make(map[string][]byte)
	for i := 0; i < 10000; i++ {
		key := fmt.Sprintf("key-%d", i)
		m[key] = make([]byte, 1024)
	}
	
	keys := make([]string, 10000)
	for i := 0; i < 10000; i++ {
		keys[i] = fmt.Sprintf("key-%d", i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m[keys[i%10000]]
	}
}

func BenchmarkSyncMapAccess(b *testing.B) {
	var m sync.Map
	for i := 0; i < 10000; i++ {
		key := fmt.Sprintf("key-%d", i)
		m.Store(key, make([]byte, 1024))
	}
	
	keys := make([]string, 10000)
	for i := 0; i < 10000; i++ {
		keys[i] = fmt.Sprintf("key-%d", i)
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			m.Load(keys[i%10000])
			i++
		}
	})
}

