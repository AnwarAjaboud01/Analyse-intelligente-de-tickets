// chatbot.js - Simple chatbot integration

document.addEventListener('DOMContentLoaded', function() {
    const chatbot = document.getElementById('chatbot');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const toggleBtn = document.getElementById('chatbot-toggle-btn');
    const closeBtn = document.getElementById('chatbot-close');

    // Toggle chatbot visibility
    function toggleChatbot() {
        chatbot.classList.toggle('show');
    }

    // Event listeners
    toggleBtn.addEventListener('click', toggleChatbot);
    closeBtn.addEventListener('click', toggleChatbot);

    // Send message
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, 'user-message');
        chatInput.value = '';

        // Call API for bot response
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = data.response || data.error || 'Erreur de réponse';
            addMessage(botResponse, 'bot-message');
        })
        .catch(error => {
            addMessage('Erreur de connexion au serveur.', 'bot-message');
            console.error('Error:', error);
        });
    }

    // Add message to chat
    function addMessage(text, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        messageDiv.innerHTML = `<strong>${className === 'user-message' ? 'Vous:' : 'Bot:'}</strong> ${text}`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Note: Bot responses are now handled by the Flask API /api/chat

    // Event listeners for sending
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Note: Bot responses are now handled by the Flask API /api/chat
});