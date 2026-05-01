// AskDoc AI - Frontend Logic
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const docInfo = document.getElementById('docInfo');
const docName = document.getElementById('docName');
const docChunks = document.getElementById('docChunks');
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const headerBadge = document.getElementById('headerBadge');

let documentLoaded = false;

// Upload handlers
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        uploadFile(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
});

async function uploadFile(file) {
    uploadArea.classList.add('uploading');
    uploadArea.querySelector('.upload-text').textContent = 'Processing...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            documentLoaded = true;
            docInfo.style.display = 'block';
            docName.textContent = data.filename;
            docChunks.textContent = `${data.chunks} searchable chunks`;
            headerBadge.textContent = data.filename;
            headerBadge.classList.add('active');
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.placeholder = 'Ask a question about your document...';

            // Clear welcome message and add system message
            chatMessages.innerHTML = '';
            addMessage('bot', `📄 **${data.filename}** loaded successfully!\n\nI've processed it into ${data.chunks} chunks. Ask me anything about the document.`);
        } else {
            addMessage('bot', `❌ Error: ${data.detail}`);
        }
    } catch (err) {
        addMessage('bot', `❌ Upload failed: ${err.message}`);
    } finally {
        uploadArea.classList.remove('uploading');
        uploadArea.querySelector('.upload-text').textContent = 'Drop PDF here';
    }
}

// Chat handlers
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message || !documentLoaded) return;

    addMessage('user', message);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Show typing indicator
    const typingEl = addTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        typingEl.remove();

        if (response.ok) {
            addMessage('bot', data.response, `${data.sources} sources used`);
        } else {
            addMessage('bot', `❌ ${data.detail}`);
        }
    } catch (err) {
        typingEl.remove();
        addMessage('bot', `❌ Error: ${err.message}`);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
});

function addMessage(role, content, meta = '') {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatar = role === 'bot' ? '🤖' : '👤';

    // Simple markdown: bold, newlines, bullets
    let html = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n- /g, '\n• ')
        .replace(/\n/g, '<br>');

    msgEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div>
            <div class="message-content"><p>${html}</p></div>
            ${meta ? `<div class="message-meta">${meta}</div>` : ''}
        </div>
    `;

    chatMessages.appendChild(msgEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgEl;
}

function addTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message bot';
    el.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
}
