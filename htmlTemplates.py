css = '''
<style>
/* Minimal, clean layout */
.chat-message {
    display: flex;
    gap: 12px;
    padding: 12px 14px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #ffffff;
    margin: 8px 0;
}
.chat-message.user { border-left: 3px solid #3b82f6; }
.chat-message.bot  { border-left: 3px solid #10b981; }

.chat-message .avatar { width: 36px; height: 36px; }
.chat-message .avatar img {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    flex: 1;
    color: #111827;
    font: 400 14px/1.55 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    white-space: pre-wrap;
}

.stForm { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #fff; }
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4042/4042171.png" alt="user">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''