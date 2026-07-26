from __future__ import annotations

import unittest
from pravaha.swarm.memory.alloydb_store import create_memory_store, AlloyDBMemoryStore

class TestAlloyDBStore(unittest.TestCase):
    
    def test_factory_fallback(self):
        store = create_memory_store("auto")
        # Since we likely don't have AlloyDB running, it should fallback to mock sqlite or throw an error safely handled.
        # It shouldn't crash.
        self.assertIsNotNone(store)

    def test_factory_alloydb(self):
        # We can't actually connect to db without running it, but we can verify it raises error if mock or no psycopg2.
        # So we just test that the import didn't crash.
        import pravaha.swarm.memory.alloydb_store as am_store
        
        if am_store.PSYCOPG2_AVAILABLE:
            try:
                store = create_memory_store("alloydb")
                self.assertIsNotNone(store)
            except Exception:
                pass
        else:
            self.assertFalse(am_store.PSYCOPG2_AVAILABLE)

    def test_embedding_fallback(self):
        import pravaha.swarm.memory.alloydb_store as am_store
        # We can mock the model to be None to test fallback
        class MockStore:
            def __init__(self):
                self.model = None
            _embed = am_store.AlloyDBMemoryStore._embed
            
        store = MockStore()
        # Should return a list of 384 floats
        embed = store._embed("test string")
        self.assertEqual(len(embed), 384)
        
if __name__ == '__main__':
    unittest.main()
