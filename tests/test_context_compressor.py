from __future__ import annotations

import unittest
from pravaha.swarm.context_compressor import ContextCompressor

class TestContextCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = ContextCompressor()
        
    def test_auto_detect_json(self):
        content = '{"key": "value"}'
        self.assertEqual(self.compressor._detect_type(content), "json_data")
        
    def test_auto_detect_build_log(self):
        content = "[INFO] Starting\n[ERROR] Failed\n[WARNING] issue"
        self.assertEqual(self.compressor._detect_type(content), "build_log")
        
    def test_auto_detect_stack_trace(self):
        content = "Traceback (most recent call last):\n  File 'test.py', line 1\nError: something went wrong"
        self.assertEqual(self.compressor._detect_type(content), "stack_trace")
        
    def test_auto_detect_source_code(self):
        content = "def hello():\n    print('world')"
        self.assertEqual(self.compressor._detect_type(content), "source_code")
        
    def test_compress_json(self):
        # generate a long json
        long_json = "{" + ", ".join(f'"{i}": "value"' for i in range(300)) + "}"
        compressed = self.compressor.compress(long_json, "json_data")
        self.assertTrue("{" in compressed)
        self.assertTrue("}" in compressed)

    def test_compress_source_code(self):
        lines = ["def foo():\n    pass\n"] * 210
        content = "".join(lines)
        compressed = self.compressor.compress(content, "source_code")
        self.assertTrue("def foo" in compressed)

    def test_compress_build_log(self):
        lines = ["[INFO] line"] * 10 + ["[ERROR] failed!"] + ["[INFO] line"] * 10
        content = "\n".join(lines)
        compressed = self.compressor.compress(content, "build_log")
        self.assertTrue("[ERROR] failed!" in compressed)
        
    def test_compress_command_output(self):
        lines = [f"output {i}" for i in range(150)]
        content = "\n".join(lines)
        compressed = self.compressor.compress(content, "command_output")
        self.assertTrue("output 0" in compressed)
        self.assertTrue("lines omitted" in compressed)
        self.assertTrue("output 149" in compressed)

if __name__ == '__main__':
    unittest.main()
