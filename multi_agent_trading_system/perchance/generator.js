// =============================================================================
// MATS Perchance generator — the thin front-end.
//
// The file is intentionally small. All key material stays on the relay; the
// Perchance iframe only holds the tunnel URL + Turnstile token and talks to
// a single POST /chat endpoint over SSE.
//
// ConduitHooks is the modular seam — Architectures 2-5 add domain routing,
// graph panels, audit overlays without rewriting this base.
// =============================================================================

const DEFAULT_TUNNEL = 'https://YOUR-TUNNEL.example.com';
const DEFAULT_SECRET = ''; // overridden per-session via localStorage

const ConduitHooks = {
  beforeSend(payload) { return payload; },
  onStream(token) {},
  onToolCall(tool, args, result) {},
  onMemCube(cube) {},
  onAuditEntry(entry) {},
  onError(err) { console.error('[MATS]', err); },
};

function getSettings() {
  return {
    tunnel: localStorage.getItem('mats.tunnel') || DEFAULT_TUNNEL,
    secret: localStorage.getItem('mats.secret') || DEFAULT_SECRET,
    domain: localStorage.getItem('mats.domain') || 'projects',
  };
}

async function sendMessage(evt) {
  if (evt) evt.preventDefault?.();
  const input = document.getElementById('in');
  if (!input) return;
  const prompt = (input.value || '').trim();
  if (!prompt) return;
  input.value = '';
  appendBubble('you', prompt);
  const settings = getSettings();
  const payload = ConduitHooks.beforeSend({
    prompt,
    domain: settings.domain,
  });
  const body = JSON.stringify(payload);
  try {
    const resp = await fetch(settings.tunnel.replace(/\/$/, '') + '/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Conduit-Secret': settings.secret,
      },
      body,
    });
    if (!resp.ok || !resp.body) {
      ConduitHooks.onError(new Error(`HTTP ${resp.status}`));
      appendBubble('bot', `⚠️ relay error: ${resp.status}`);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    const bubble = appendBubble('bot', '');
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split('\n\n');
      buf = events.pop() || '';
      for (const event of events) {
        const line = event.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          const token = parsed?.choices?.[0]?.delta?.content || '';
          if (token) {
            ConduitHooks.onStream(token);
            bubble.textContent += token;
          }
        } catch (_) {
          /* non-JSON keepalive */
        }
      }
    }
  } catch (err) {
    ConduitHooks.onError(err);
    appendBubble('bot', `⚠️ ${err.message}`);
  }
}

function appendBubble(who, text) {
  const container = document.getElementById('log') || document.body;
  const node = document.createElement('div');
  node.className = 'bubble ' + who;
  node.textContent = text;
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
  return node;
}

async function streamEvents() {
  const settings = getSettings();
  const url = settings.tunnel.replace(/\/$/, '') + '/events';
  const source = new EventSource(url + '?token=' + encodeURIComponent(settings.secret));
  source.addEventListener('message', (ev) => {
    try {
      const parsed = JSON.parse(ev.data);
      ConduitHooks.onMemCube(parsed);
    } catch (err) {
      ConduitHooks.onError(err);
    }
  });
  source.addEventListener('error', (err) => ConduitHooks.onError(err));
}

window.MATS = { sendMessage, ConduitHooks, streamEvents, getSettings };
