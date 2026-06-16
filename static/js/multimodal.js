/**
 * 多模态聊天前端脚本 (v2)
 * ==========================
 *
 * 功能：
 * 1. 文本消息发送（含超时控制、请求取消）
 * 2. 语音录制和识别
 * 3. 摄像头拍照和情绪识别
 * 4. 语音回复播放（TTS）
 * 5. 暗色模式切换
 */

const API_BASE = '/api/v1';
const REQUEST_TIMEOUT_MS = 60000;

let sessionId = null;
let isLoading = false;
let ttsEnabled = true;
let activeAbortController = null;

let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = null;
let recordingTimer = null;

let cameraStream = null;

// ============================================================
// DOM 引用
// ============================================================
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const sessionStatus = document.getElementById('session-status');
const newSessionBtn = document.getElementById('new-session-btn');

const voiceBtn = document.getElementById('voice-btn');
const cameraBtn = document.getElementById('camera-btn');
const ttsToggle = document.getElementById('tts-toggle');
const recordingOverlay = document.getElementById('recording-overlay');
const stopRecordingBtn = document.getElementById('stop-recording-btn');
const recordingTimerDisplay = document.getElementById('recording-timer');
const recordingText = document.getElementById('recording-text');

const emotionPanel = document.getElementById('emotion-panel');
const closeEmotionBtn = document.getElementById('close-emotion-btn');
const emotionType = document.getElementById('emotion-type');
const intensityFill = document.getElementById('intensity-fill');
const intensityValue = document.getElementById('intensity-value');
const therapyStage = document.getElementById('therapy-stage');

const cameraPreview = document.getElementById('camera-preview');
const cameraCanvas = document.getElementById('camera-canvas');
const imageInput = document.getElementById('image-input');

const emotionCNMap = {
    'angry': '愤怒',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心',
    'sad': '悲伤',
    'surprise': '惊讶',
    'neutral': '平静',
    'unknown': '未知'
};

const stageCNMap = {
    'rapport_building': '建立关系',
    'assessment': '评估问题',
    'goal_setting': '目标设定',
    'intervention': '干预治疗',
    'termination': '结束阶段'
};

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    updateTTSToggle();
    initDarkMode();
});

function initEventListeners() {
    sendBtn.addEventListener('click', () => sendMessage());

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', autoResizeTextarea);

    newSessionBtn.addEventListener('click', createNewSession);

    voiceBtn.addEventListener('click', toggleVoiceRecording);
    stopRecordingBtn.addEventListener('click', stopRecording);

    cameraBtn.addEventListener('click', toggleCamera);

    ttsToggle.addEventListener('click', toggleTTS);

    closeEmotionBtn.addEventListener('click', () => {
        emotionPanel.classList.remove('show');
    });

    imageInput.addEventListener('change', handleImageUpload);

    // 暗色模式切换
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleDarkMode);
    }
}

// ============================================================
// 暗色模式
// ============================================================
function initDarkMode() {
    // 从 localStorage 读取用户偏好
    const saved = localStorage.getItem('theme');
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.body.classList.add('dark');
    }
    updateThemeToggleIcon();

    // 监听系统主题变化
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            document.body.classList.toggle('dark', e.matches);
            updateThemeToggleIcon();
        }
    });
}

function toggleDarkMode() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeToggleIcon();
}

function updateThemeToggleIcon() {
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
    }
}

// ============================================================
// 工具函数
// ============================================================
function htmlEscape(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

/** 判断用户是否在查看历史消息（距底部 > 100px），避免强制滚动 */
function shouldAutoScroll() {
    const threshold = 120;
    return chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight < threshold;
}

function scrollChat() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'auto'
    });
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

function disableInput(disabled) {
    isLoading = disabled;
    sendBtn.disabled = disabled;
    messageInput.disabled = disabled;
}

function showTyping() {
    typingIndicator.classList.add('show');
    if (shouldAutoScroll()) scrollChat();
}

function hideTyping() {
    typingIndicator.classList.remove('show');
}

function updateSessionStatus() {
    if (sessionId) {
        sessionStatus.textContent = `会话: ${sessionId.substring(0, 8)}...`;
    } else {
        sessionStatus.textContent = '会话: 未开始';
    }
}

/** 创建带超时的 fetch，超时或手动取消时 abort */
function fetchWithTimeout(url, options = {}, timeout = REQUEST_TIMEOUT_MS) {
    activeAbortController = new AbortController();
    const timer = setTimeout(() => activeAbortController.abort(), timeout);

    return fetch(url, {
        ...options,
        signal: activeAbortController.signal
    }).finally(() => clearTimeout(timer));
}

function cancelCurrentRequest() {
    if (activeAbortController) {
        activeAbortController.abort();
        activeAbortController = null;
    }
}

// ============================================================
// 消息渲染
// ============================================================
function addMessage(content, isUser, audioBase64 = null) {
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🤖';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = htmlEscape(content);

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
    });

    messageContent.appendChild(timeDiv);

    if (!isUser && audioBase64) {
        const audioPlayer = document.createElement('div');
        audioPlayer.className = 'audio-player';
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = `data:audio/mpeg;base64,${audioBase64}`;
        audioPlayer.appendChild(audio);
        messageContent.appendChild(audioPlayer);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    chatMessages.appendChild(messageDiv);
    if (shouldAutoScroll()) scrollChat();
}

function showError(content) {
    hideTyping();
    addMessage(content, false);
}

// ============================================================
// 核心：发送消息（文本 | 语音转文字后统一调用）
// ============================================================
async function sendMessage(messageOverride) {
    const message = messageOverride || messageInput.value.trim();
    if (!message || isLoading) return;

    // 如果是手动输入，先添加用户消息
    if (!messageOverride) {
        addMessage(message, true);
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }

    disableInput(true);
    showTyping();

    try {
        const response = await fetchWithTimeout(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: sessionId,
                enable_thought_chain: true,
                enable_emotion_analysis: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        activeAbortController = null;

        sessionId = data.session_id;
        updateSessionStatus();

        hideTyping();

        // TTS
        let audioBase64 = null;
        if (ttsEnabled && data.response) {
            audioBase64 = await fetchTTS(data.response);
        }

        addMessage(data.response, false, audioBase64);

        if (data.emotion_analysis) {
            updateEmotionPanel(data.emotion_analysis);
        }

    } catch (error) {
        activeAbortController = null;
        if (error.name === 'AbortError') {
            showError('请求超时，请检查网络后重试。');
        } else {
            console.error('Chat error:', error);
            showError('抱歉，发生了错误。请稍后重试。');
        }
    } finally {
        disableInput(false);
    }
}

async function fetchTTS(text) {
    try {
        const response = await fetch(`${API_BASE}/multimodal/text-to-speech/base64`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const data = await response.json();
            return data.audio_base64;
        }
    } catch (e) {
        console.warn('TTS failed:', e);
    }
    return null;
}

// ============================================================
// 语音录制
// ============================================================
async function toggleVoiceRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await handleSpeechToText(audioBlob);

            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();

        voiceBtn.classList.add('recording');
        recordingOverlay.classList.add('show');

        recordingTimer = setInterval(updateRecordingTimer, 1000);

    } catch (error) {
        console.error('Error starting recording:', error);
        alert('无法访问麦克风，请检查权限设置。');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }

    voiceBtn.classList.remove('recording');
    recordingOverlay.classList.remove('show');

    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
}

function updateRecordingTimer() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    recordingTimerDisplay.textContent = `${mins}:${secs}`;
}

async function handleSpeechToText(audioBlob) {
    disableInput(true);
    showTyping();
    recordingText.textContent = '正在识别语音...';

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch(`${API_BASE}/multimodal/speech-to-text`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (data.text) {
            addMessage(`🎤 ${data.text}`, true);
            // 统一使用 sendMessage 发送
            await sendMessage(data.text);
        } else {
            hideTyping();
            addMessage('未能识别语音内容，请重试。', false);
        }

    } catch (error) {
        console.error('STT error:', error);
        hideTyping();
        addMessage('语音识别失败，请重试。', false);
    } finally {
        disableInput(false);
        recordingText.textContent = '正在录音...';
    }
}

// ============================================================
// 摄像头 & 情绪识别
// ============================================================
async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });

        cameraPreview.srcObject = cameraStream;
        cameraPreview.classList.add('show');
        cameraBtn.classList.add('active');

        cameraPreview.addEventListener('click', captureAndAnalyze);

    } catch (error) {
        console.error('Error starting camera:', error);
        alert('无法访问摄像头，请检查权限设置。');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraPreview.classList.remove('show');
    cameraBtn.classList.remove('active');
}

async function captureAndAnalyze() {
    if (!cameraStream) return;

    const context = cameraCanvas.getContext('2d');
    cameraCanvas.width = cameraPreview.videoWidth;
    cameraCanvas.height = cameraPreview.videoHeight;
    context.drawImage(cameraPreview, 0, 0);

    const imageData = cameraCanvas.toDataURL('image/jpeg', 0.8);
    const base64Data = imageData.split(',')[1];

    disableInput(true);
    showTyping();

    try {
        const response = await fetch(`${API_BASE}/multimodal/emotion-detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: base64Data })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        hideTyping();

        if (data.face_detected) {
            updateEmotionPanel({
                primary_emotion: data.primary_emotion,
                intensity: data.confidence
            });

            addMessage(`📷 检测到情绪: ${emotionCNMap[data.primary_emotion]} (${Math.round(data.confidence * 100)}%)`, false);
        } else {
            addMessage('📷 未检测到人脸，请确保面部在画面中。', false);
        }

    } catch (error) {
        console.error('Emotion detect error:', error);
        hideTyping();
        addMessage('情绪识别失败，请重试。', false);
    } finally {
        disableInput(false);
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
        const base64Data = e.target.result.split(',')[1];

        disableInput(true);
        showTyping();

        try {
            const response = await fetch(`${API_BASE}/multimodal/emotion-detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: base64Data })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            hideTyping();

            if (data.face_detected) {
                updateEmotionPanel({
                    primary_emotion: data.primary_emotion,
                    intensity: data.confidence
                });

                addMessage(`📷 检测到情绪: ${emotionCNMap[data.primary_emotion]} (${Math.round(data.confidence * 100)}%)`, false);
            } else {
                addMessage('📷 未检测到人脸，请确保图片中有清晰的面部。', false);
            }

        } catch (error) {
            console.error('Emotion detect error:', error);
            hideTyping();
            addMessage('情绪识别失败，请重试。', false);
        } finally {
            disableInput(false);
        }
    };

    reader.readAsDataURL(file);
    // 重置 input 以允许重复上传同一文件
    event.target.value = '';
}

// ============================================================
// 会话 & 情绪面板 & TTS
// ============================================================
async function createNewSession() {
    cancelCurrentRequest();
    sessionId = null;
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-illustration">
                <div class="welcome-halo"></div>
                <div class="welcome-icon">💬</div>
            </div>
            <h2>欢迎使用心理咨询AI助手</h2>
            <p>我是一个专业的心理咨询AI，可以倾听你的烦恼，帮助你缓解压力。</p>
            <p>请放心，我们的对话是保密的。你可以随时开始分享你的感受。</p>
            <div class="feature-tags">
                <span class="tag">🎤 语音输入</span>
                <span class="tag">📷 情绪识别</span>
                <span class="tag">🔊 语音回复</span>
            </div>
        </div>
    `;
    updateSessionStatus();
    emotionPanel.classList.remove('show');
    messageInput.focus();
}

function updateEmotionPanel(emotionState) {
    emotionPanel.classList.add('show');

    const primaryEmotion = emotionState.primary_emotion || 'unknown';
    const intensity = Math.round((emotionState.intensity || 0.5) * 100);
    const stage = emotionState.therapy_stage || 'rapport_building';

    emotionType.textContent = emotionCNMap[primaryEmotion] || primaryEmotion;
    intensityFill.style.width = intensity + '%';
    intensityValue.textContent = intensity + '%';
    therapyStage.textContent = stageCNMap[stage] || stage;
}

function toggleTTS() {
    ttsEnabled = !ttsEnabled;
    updateTTSToggle();
}

function updateTTSToggle() {
    ttsToggle.classList.toggle('active', ttsEnabled);
}
