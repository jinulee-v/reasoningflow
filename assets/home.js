(() => {
    document.documentElement.classList.add('js');
    const shell = document.querySelector('.video-shell');
    const youtubeId = shell && shell.dataset.youtubeId
        ? shell.dataset.youtubeId.trim()
        : '';
    if (shell && youtubeId) {
        const iframe = document.createElement('iframe');
        iframe.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(youtubeId)}?rel=0`;
        iframe.title = 'ReasoningFlow conference presentation';
        iframe.loading = 'lazy';
        iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
        iframe.allowFullscreen = true;
        shell.replaceChildren(iframe);
        shell.classList.add('has-video');
    }
    const reveals = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) {
        reveals.forEach((item) => item.classList.add('is-visible'));
        return;
    }
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });
    reveals.forEach((item) => observer.observe(item));
})();
