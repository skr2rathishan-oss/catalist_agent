"""Explicit opt-in check of real credentials and NVIDIA's embedding endpoint."""

import os
import unittest

from app.llm import NvidiaEmbeddingClient


@unittest.skipUnless(os.environ.get("RUN_LIVE_EMBEDDING_TESTS") == "1", "Set RUN_LIVE_EMBEDDING_TESTS=1 to use NVIDIA API quota")
class TestNvidiaLive(unittest.TestCase):
    def test_document_and_query_embeddings(self):

        client = NvidiaEmbeddingClient()
        passage = client.embed(["The team chose a battery to power the remote control."], task="document")[0]
        query = client.embed(["What power source did the team choose?"], task="query")[0]
        self.assertEqual(len(passage), client.settings.dimensions)
        self.assertEqual(len(query), client.settings.dimensions)
        self.assertAlmostEqual(sum(value * value for value in passage), 1.0, places=5)
        self.assertAlmostEqual(sum(value * value for value in query), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
