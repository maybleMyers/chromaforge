// Keeps the ForgeCanvas toolbar inside the visible area of its canvas.
//
// maximize() pins the canvas container to the layout viewport (position: fixed,
// 100vw/100vh). A precision-touchpad pinch zooms the *visual* viewport instead,
// which leaves the container -- and the toolbar sitting in its top-left corner --
// outside what is actually on screen until you pan back. The same thing happens
// unmaximized when the canvas is taller than the window and the page is scrolled.
//
// This slides the toolbar into the intersection of its container and the visual
// viewport. It works off the DOM alone, so it stays correct if canvas.min.js
// (which is obfuscated and must not be edited) is ever rebuilt upstream.

(function () {
    'use strict';

    var queued = false;

    function reposition() {
        queued = false;

        var vv = window.visualViewport;
        var vLeft = vv ? vv.offsetLeft : 0;
        var vTop = vv ? vv.offsetTop : 0;

        var toolbars = document.querySelectorAll('.forge-toolbar, .forge-toolbar-static');

        for (var i = 0; i < toolbars.length; i++) {
            var tb = toolbars[i];
            var box = tb.parentElement;             // .forge-image-container(-plain)
            if (!box) continue;

            var r = box.getBoundingClientRect();
            if (!r.width || !r.height) continue;    // canvas on a hidden tab

            // Where the visual viewport's top-left lands, in container coordinates.
            var left = vLeft - r.left;
            var top = vTop - r.top;

            // Never leave the container's own box.
            left = Math.max(0, Math.min(left, r.width - tb.offsetWidth));
            top = Math.max(0, Math.min(top, r.height - tb.offsetHeight));

            tb.style.left = left + 'px';
            tb.style.top = top + 'px';

            // maximize() signals fullscreen with an inline position:fixed on .forge-container.
            var container = box.parentElement;
            if (container) {
                container.classList.toggle('forge-maximized', container.style.position === 'fixed');
            }

            observe(box);
        }
    }

    function schedule() {
        if (!queued) {
            queued = true;
            requestAnimationFrame(reposition);
        }
    }

    // Re-run when a canvas is resized via the .forge-resize-line drag.
    var resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(schedule) : null;
    var observed = typeof WeakSet === 'function' ? new WeakSet() : null;

    function observe(box) {
        if (!resizeObserver || !observed || observed.has(box)) return;
        observed.add(box);
        resizeObserver.observe(box);
    }

    window.addEventListener('scroll', schedule, {passive: true});
    window.addEventListener('resize', schedule, {passive: true});

    if (window.visualViewport) {
        window.visualViewport.addEventListener('scroll', schedule, {passive: true});
        window.visualViewport.addEventListener('resize', schedule, {passive: true});
    }

    // Maximize/minimize take effect immediately rather than on the next scroll.
    document.addEventListener('click', schedule, true);

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', schedule);
    } else {
        schedule();
    }
})();
