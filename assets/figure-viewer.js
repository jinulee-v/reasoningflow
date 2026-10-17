(() => {
    const root = document.querySelector("[data-figure-viewer]");
    if (!root) return;

    const viewport = root.querySelector(".figure-viewport");
    const image = viewport.querySelector("img");
    const zoomIn = root.querySelector("[data-zoom-in]");
    const zoomOut = root.querySelector("[data-zoom-out]");
    const resetButton = root.querySelector("[data-reset]");

    const state = { scale: 1, x: 0, y: 0, pointerId: null, startX: 0, startY: 0 };
    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

    function render() {
        image.style.transform = `translate(-50%, -50%) translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    }

    function setScale(nextScale, originX = viewport.clientWidth / 2, originY = viewport.clientHeight / 2) {
        const previousScale = state.scale;
        state.scale = clamp(nextScale, 1, 4);
        const ratio = state.scale / previousScale;
        state.x = originX - viewport.clientWidth / 2 - (originX - viewport.clientWidth / 2 - state.x) * ratio;
        state.y = originY - viewport.clientHeight / 2 - (originY - viewport.clientHeight / 2 - state.y) * ratio;
        if (state.scale === 1) {
            state.x = 0;
            state.y = 0;
        }
        render();
    }

    function reset() {
        state.scale = 1;
        state.x = 0;
        state.y = 0;
        render();
    }

    viewport.addEventListener("wheel", (event) => {
        event.preventDefault();
        const bounds = viewport.getBoundingClientRect();
        const factor = event.deltaY < 0 ? 1.14 : 1 / 1.14;
        setScale(state.scale * factor, event.clientX - bounds.left, event.clientY - bounds.top);
    }, { passive: false });

    viewport.addEventListener("pointerdown", (event) => {
        state.pointerId = event.pointerId;
        state.startX = event.clientX - state.x;
        state.startY = event.clientY - state.y;
        viewport.setPointerCapture(event.pointerId);
        viewport.classList.add("is-dragging");
    });

    viewport.addEventListener("pointermove", (event) => {
        if (event.pointerId !== state.pointerId) return;
        state.x = event.clientX - state.startX;
        state.y = event.clientY - state.startY;
        render();
    });

    function finishDrag(event) {
        if (event.pointerId !== state.pointerId) return;
        state.pointerId = null;
        viewport.classList.remove("is-dragging");
    }

    viewport.addEventListener("pointerup", finishDrag);
    viewport.addEventListener("pointercancel", finishDrag);

    viewport.addEventListener("keydown", (event) => {
        const movement = event.shiftKey ? 40 : 18;
        if (["+", "=", "-", "0", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
            event.preventDefault();
        }
        if (event.key === "+" || event.key === "=") setScale(state.scale * 1.2);
        if (event.key === "-") setScale(state.scale / 1.2);
        if (event.key === "0") reset();
        if (event.key === "ArrowLeft") state.x += movement;
        if (event.key === "ArrowRight") state.x -= movement;
        if (event.key === "ArrowUp") state.y += movement;
        if (event.key === "ArrowDown") state.y -= movement;
        render();
    });

    zoomIn.addEventListener("click", () => setScale(state.scale * 1.25));
    zoomOut.addEventListener("click", () => setScale(state.scale / 1.25));
    resetButton.addEventListener("click", reset);
    window.addEventListener("resize", reset);
    render();
})();
