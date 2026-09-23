import asyncio
import json
import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import api


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.data = api.graph_data()
        self.gid = self.data['nodes'][0]['gid']

    def test_cluster_roles_and_cutoff(self):
        for cluster in self.data['clusters']:
            members = [n for n in self.data['nodes'] if n['cluster_id'] == cluster['cluster_id']]
            self.assertEqual(dict(Counter(n['role'] for n in members)), cluster['role_counts'])
            self.assertEqual(len(members), cluster['n_nodes'])
        for node in self.data['nodes']:
            self.assertTrue(50 <= len(node['evidence']) <= 200)
            self.assertIn('₸', node['evidence'])
            self.assertNotIn('KZT', node['evidence'])
            self.assertNotRegex(node['evidence'], r'\d,\d{3}')
            self.assertFalse(node['is_cutoff'] and node['role'] == 'terminal')
            self.assertFalse(node['is_seed'] and node['role'] in ('terminal', 'transit'))

    def test_selection_and_cors(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            r = self.client.post('/api/ask', json={'question': 'inspect', 'selected_gids': [str(self.gid)]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['steps'][0]['args']['gid'], self.gid)
        self.assertIn(self.gid, r.json()['highlight_gids'])
        self.assertEqual(r.json()['llm_error'], 'not_configured')
        r = self.client.options('/api/ask', headers={'Origin': 'http://localhost:4174', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
        self.assertEqual(r.headers['access-control-allow-origin'], 'http://localhost:4174')
        self.assertEqual(self.client.post('/api/ask', json={'question': 'x', 'selected_gids': [-1]}).status_code, 422)

    def test_function_call_roundtrip(self):
        requests = []
        gid = self.gid
        class Message:
            content = None
            tool_calls = [SimpleNamespace(id='call_1', function=SimpleNamespace(name='get_node', arguments=json.dumps({'gid': gid})))]
            def model_dump(self, **kwargs):
                return {'role': 'assistant', 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'get_node', 'arguments': json.dumps({'gid': gid})}}]}
        class FakeClient:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def create(self, **kwargs):
                requests.append(kwargs)
                msg = Message() if len(requests) == 1 else SimpleNamespace(tool_calls=None, content=f'Hypothesis gid {gid}')
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}), patch('openai.AsyncOpenAI', FakeClient):
            response = self.client.post('/api/ask', json={'question': 'inspect selected', 'selected_gids': [str(gid)]}).json()
        self.assertEqual(response['mode'], 'llm')
        self.assertEqual(requests[0]['tool_choice'], 'required')
        self.assertIn(str(gid), requests[0]['messages'][1]['content'])
        self.assertEqual(response['steps'][0]['tool'], 'get_node')
        self.assertGreater(len(response['highlight_gids']), 1)

    def test_timeout_keeps_evidence(self):
        async def slow(question, gids, steps, touched):
            result = api.tool('get_node', {'gid': self.gid})
            steps.append({'tool': 'get_node', 'args': {'gid': self.gid}, 'result': result})
            touched.update(api.collect_gids(result))
            await asyncio.sleep(10)
        original = asyncio.wait_for
        async def short(awaitable, timeout):
            return await original(awaitable, timeout=0.05)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}), patch('api.model_answer', slow), patch('api.asyncio.wait_for', short):
            result = self.client.post('/api/ask', json={'question': 'inspect', 'selected_gids': [str(self.gid)]}).json()
        self.assertEqual(result['mode'], 'fallback')
        self.assertEqual(result['llm_error'], 'TimeoutError')
        self.assertEqual(result['steps'][0]['tool'], 'get_node')


if __name__ == '__main__':
    unittest.main()
