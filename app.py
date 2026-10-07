import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "gemini").lower()  # gemini | deepseek
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "তুমি একজন বন্ধুসুলভ সহকারী। ব্যবহারকারী যে ভাষায় লেখে সেই ভাষাতেই উত্তর দাও।",
)

app = FastAPI(title="AI Chatbot")


class Message(BaseModel):
    role: str  # "user" অথবা "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


# ------------------------------------------------------------ Gemini মোড
_gemini = None


def stream_gemini(messages):
    global _gemini
    from google import genai
    from google.genai import types

    if _gemini is None:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY সেট করা নেই")
        _gemini = genai.Client(api_key=key)

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part(text=m.content)],
        )
        for m in messages
    ]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

    for chunk in _gemini.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        if chunk.text:
            yield chunk.text


# ----------------------------------------------------------- DeepSeek মোড
_deepseek = None


def stream_deepseek(messages):
    global _deepseek
    from openai import OpenAI

    if _deepseek is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY সেট করা নেই")
        _deepseek = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat += [{"role": m.role, "content": m.content} for m in messages]
    stream = _deepseek.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), messages=chat, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ------------------------------------------------------------------ রুট
@app.post("/api/chat")
def chat(req: ChatRequest):
    gen = stream_deepseek if PROVIDER == "deepseek" else stream_gemini

    def event_stream():
        try:
            for piece in gen(req.messages):
                yield f"data: {json.dumps({'text': piece})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


INDEX_HTML = r'''<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI চ্যাটবট</title>
<style>
  :root { --bg:#f5f6fa; --card:#fff; --text:#1f2430; --muted:#6b7280; --accent:#4f6ef7; --user:#4f6ef7; --bot:#eef0f6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12141a; --card:#1b1e27; --text:#e8eaf0; --muted:#8b93a7; --bot:#262a36; }
  }
  * { box-sizing:border-box; }
  html,body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,"Noto Sans Bengali","Kalpurush",sans-serif; display:flex; justify-content:center; }
  .app { width:100%; max-width:820px; height:100%; display:flex; flex-direction:column; background:var(--card); }
  header { padding:14px 18px; border-bottom:1px solid var(--bot); display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size:18px; margin:0; }
  header small { color:var(--muted); }
  button { font:inherit; cursor:pointer; }
  #clear { background:none; border:1px solid var(--muted); color:var(--muted); border-radius:8px; padding:6px 12px; }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:85%; padding:10px 14px; border-radius:14px; line-height:1.6; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:var(--user); color:#fff; border-bottom-right-radius:4px; }
  .bot { align-self:flex-start; background:var(--bot); border-bottom-left-radius:4px; }
  .err { color:#e5484d; }
  form { display:flex; gap:10px; padding:14px 18px; border-top:1px solid var(--bot); }
  textarea { flex:1; resize:none; max-height:140px; padding:10px 12px; border-radius:12px; border:1px solid var(--muted); background:var(--card); color:var(--text); font:inherit; }
  #send { background:var(--accent); color:#fff; border:0; border-radius:12px; padding:0 20px; }
  #send:disabled { opacity:.5; cursor:not-allowed; }
  .hint { text-align:center; color:var(--muted); margin:auto; }
</style>
</head>
<body>
<div class="app">
  <header>
    <div><h1>AI চ্যাটবট</h1><small>Google Gemini দ্বারা চালিত</small></div>
    <button id="clear" type="button">নতুন চ্যাট</button>
  </header>
  <div id="log"><div class="hint" id="hint">কিছু লিখে শুরু করুন 👇</div></div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="আপনার বার্তা লিখুন..." autofocus></textarea>
    <button id="send" type="submit">পাঠান</button>
  </form>
</div>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const send = document.getElementById('send');
let history = [];

function add(role, text) {
  const hint = document.getElementById('hint'); if (hint) hint.remove();
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = input.scrollHeight + 'px'; });

document.getElementById('clear').onclick = () => {
  history = []; log.innerHTML = '<div class="hint" id="hint">কিছু লিখে শুরু করুন 👇</div>';
};

form.addEventListener('submit', async e => {
  e.preventDefault();
  const text = input.value.trim(); if (!text) return;
  input.value = ''; input.style.height = 'auto';
  add('user', text); history.push({ role: 'user', content: text });
  const bubble = add('bot', '...'); let answer = '';
  send.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history })
    });
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n'); buf = parts.pop();
      for (const p of parts) {
        const data = p.replace(/^data: /, '');
        if (data === '[DONE]') continue;
        const j = JSON.parse(data);
        if (j.error) { bubble.classList.add('err'); bubble.textContent = 'ত্রুটি: ' + j.error; }
        else { answer += j.text; bubble.textContent = answer; log.scrollTop = log.scrollHeight; }
      }
    }
    if (answer) history.push({ role: 'assistant', content: answer });
    else history.pop();
  } catch (err) {
    bubble.classList.add('err'); bubble.textContent = 'সার্ভারে সংযোগ করা যায়নি।'; history.pop();
  }
  send.disabled = false; input.focus();
});
</script>
</body>
</html>
'''


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
