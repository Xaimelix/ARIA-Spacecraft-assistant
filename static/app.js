// Upload form handler
document.addEventListener('DOMContentLoaded', () => {
  const uploadForm = document.getElementById('upload-form');
  const uploadResult = document.getElementById('upload-result');
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      uploadResult.textContent = 'Загрузка…';
      const formData = new FormData(uploadForm);
      try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();
        uploadResult.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        uploadResult.textContent = 'Ошибка: ' + String(err);
      }
    });
  }

  // Chat handler
  const chatForm = document.getElementById('chat-form');
  const chatLog = document.getElementById('chat-log');
  const chatMsg = document.getElementById('chat-message');
  if (chatForm && chatLog && chatMsg) {
    const append = (role, text) => {
      const div = document.createElement('div');
      div.className = 'bubble ' + (role === 'user' ? 'user' : 'bot');
      div.textContent = text;
      chatLog.appendChild(div);
      chatLog.scrollTop = chatLog.scrollHeight;
      return div;
    };

    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = chatMsg.value.trim();
      if (!message) return;
      append('user', message);
      chatMsg.value = '';
      const pending = append('bot', 'Думаю…');
      try {
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const data = await resp.json();
        pending.textContent = data.ok ? data.answer : (data.error || 'Ошибка');
        if (data.snippets && data.snippets.length) {
          data.snippets.forEach(s => {
            const sn = document.createElement('div');
            sn.className = 'snippet';
            sn.textContent = `[${s.doc_id || 'doc'}] ${s.preview}`;
            chatLog.appendChild(sn);
          });
        }
      } catch (err) {
        pending.textContent = 'Ошибка: ' + String(err);
      }
      chatLog.scrollTop = chatLog.scrollHeight;
    });
  }
});


