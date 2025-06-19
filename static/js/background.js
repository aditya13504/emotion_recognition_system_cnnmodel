document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('background-canvas');
    if (!canvas) {
        console.error('Background canvas not found!');
        return;
    }
    const ctx = canvas.getContext('2d');

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    let stars = [];
    const numStars = 200; // Adjust density of stars
    const speed = 1; // Adjust speed of stars

    // Star object
    function Star(x, y, size, velocity) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.velocity = velocity;
    }

    // Initialize stars
    function init() {
        stars = [];
        for (let i = 0; i < numStars; i++) {
            const x = Math.random() * width;
            const y = Math.random() * height;
            const size = Math.random() * 1.5 + 0.5; // Star size
            const angle = Math.atan2(height, width); // Diagonal angle (bottom-right)
            const velocity = {
                x: Math.cos(angle) * (Math.random() * 0.5 + 0.2) * speed, // Add slight random variation to speed
                y: Math.sin(angle) * (Math.random() * 0.5 + 0.2) * speed
            };
            stars.push(new Star(x, y, size, velocity));
        }
    }

    // Draw stars
    function draw() {
        ctx.clearRect(0, 0, width, height); // Clear canvas
        ctx.fillStyle = 'white';

        stars.forEach(star => {
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // Update star positions
    function update() {
        stars.forEach(star => {
            star.x += star.velocity.x;
            star.y += star.velocity.y;

            // Reset stars that go off-screen
            if (star.x > width + star.size) {
                star.x = -star.size;
                star.y = Math.random() * height; // Reappear at random height on left
            }
            if (star.y > height + star.size) {
                star.y = -star.size;
                star.x = Math.random() * width; // Reappear at random width on top
            }
        });
    }

    // Animation loop
    function animate() {
        draw();
        update();
        requestAnimationFrame(animate);
    }

    // Handle window resize
    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        init(); // Re-initialize stars on resize
    });

    // Start
    init();
    animate();
}); 