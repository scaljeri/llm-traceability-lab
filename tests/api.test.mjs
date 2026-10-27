import {test} from 'node:test';
import assert from 'node:assert/strict';
import {api} from '../src/traceability_lab/web/api.mjs';

test('API keeps structured validation errors and the HTTP status', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({detail:[{msg:'Invalid value'}]}), {status:422}));
  await assert.rejects(api('/test', {}), error => error.status === 422 && error.message.includes('Invalid value'));
});
test('API reports an HTML server error as an HTTP error', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('<html>Bad Gateway</html>', {status:502, statusText:'Bad Gateway'}));
  await assert.rejects(api('/test'), error => error.status === 502 && error.message === 'HTTP 502 Bad Gateway');
});
test('API accepts successful empty responses and handles empty errors', async t => {
  const mock = t.mock.method(globalThis, 'fetch', async () => new Response(null, {status:204}));
  assert.equal(await api('/test'), null);
  mock.mock.mockImplementation(async () => new Response(null, {status:503}));
  await assert.rejects(api('/test'), error => error.status === 503 && error.message === 'HTTP 503');
});
test('API rejects malformed JSON on success and preserves normal payloads', async t => {
  const mock = t.mock.method(globalThis, 'fetch', async () => new Response('{bad', {status:200}));
  await assert.rejects(api('/test'), /Invalid JSON response/);
  mock.mock.mockImplementation(async (_url, options) => new Response(options.body));
  assert.deepEqual(await api('/test', {value:42}), {value:42});
});
