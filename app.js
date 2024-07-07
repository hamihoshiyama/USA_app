// ページがロードされたときに通知の権限を確認する
document.addEventListener('DOMContentLoaded', () => {
    if (Notification.permission !== 'granted') {
        Notification.requestPermission();
    }
});

document.getElementById('recordButton').addEventListener('click', () => {
    recordAudio();
});

// 音声を録音して送信
function recordAudio() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.ondataavailable = (e) => {
                const formData = new FormData();
                formData.append('file', e.data, 'audio.wav');

                fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('response').textContent = data.response;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            };

            setTimeout(() => {
                mediaRecorder.stop();
            }, 5000);  // 5秒間録音
        });
}