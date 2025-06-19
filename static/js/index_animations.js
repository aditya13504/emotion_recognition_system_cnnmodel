// --- Floating Emoji Logic for Index Page ---
document.addEventListener('DOMContentLoaded', () => {
    const emojiContainer = document.getElementById('floating-emojis-container');
    if (!emojiContainer) {
        console.warn("Emoji container not found on this page.");
        return; // Exit if container doesn't exist
    }

    // List of emojis to use (can include others)
    const emojis = ['😄', '😢', '😠', '😨', '🤢', '😮', '😊', '🥳', '🤩', '😭', '😡'];
    const emojiInterval = 300; // Milliseconds between new emojis

    function createFloatingEmoji() {
        const emoji = document.createElement('span');
        emoji.classList.add('floating-emoji');
        emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];

        // Random horizontal position
        emoji.style.left = `${Math.random() * 95}vw`; // Use vw for viewport width

        // Optional: Randomize animation duration slightly
        const duration = Math.random() * 2 + 4; // Between 4 and 6 seconds
        emoji.style.animationDuration = `${duration}s`;

        // Optional: Randomize font size slightly
        const size = Math.random() * 1 + 1.5; // Between 1.5rem and 2.5rem
        emoji.style.fontSize = `${size}rem`;

        emojiContainer.appendChild(emoji);

        // Remove emoji after animation ends (duration + small buffer)
        setTimeout(() => {
            emoji.remove();
        }, duration * 1000 + 100);
    }

    // Start creating emojis
    setInterval(createFloatingEmoji, emojiInterval);
}); 