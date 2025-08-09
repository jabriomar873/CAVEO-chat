css = '''
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 10px; 
    margin-bottom: 1rem; 
    display: flex;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.chat-message.user {
    background: linear-gradient(135deg, #4287f5 0%, #2563eb 100%);
}
.chat-message.bot {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}
.chat-message .avatar {
    width: 15%;
    min-width: 50px;
}
.chat-message .avatar img {
    width: 45px;
    height: 45px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(255,255,255,0.3);
}
.chat-message .message {
    width: 85%;
    padding: 0 1rem;
    color: #fff;
    font-size: 14px;
    line-height: 1.5;
}
.stForm {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e9ecef;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4042/4042171.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''