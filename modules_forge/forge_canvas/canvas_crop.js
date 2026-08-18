// Drag-to-crop for ForgeCanvas.
//
// canvas.min.js is obfuscated and must not be edited, so this works off the DOM alone -- the same
// constraint canvas_keep_toolbar.js is written under. Two properties of that file make it possible:
//
//   * pan/zoom is plain inline geometry. drawImage() sets style.left/top/width/height on both
//     #image_UUID and #drawingCanvas_UUID, so screen -> source pixels is derivable from the DOM and
//     stays correct while panned, zoomed or maximized.
//   * GradioTextAreaBind.listen() polls its textarea's .value every 100ms, so writing a data URL
//     into the hidden background/foreground textbox and firing a bubbling 'input' event updates both
//     the rendered canvas and Gradio's server-side state. That is the same channel the existing
//     "copy to inpaint" buttons already push images through.

(function () {
    'use strict';

    // Cropping only makes sense on the four img2img source canvases. ControlNet builds its own
    // ForgeCanvas instances from the same canvas.html, and its preview canvas is read-only.
    var CROP_TARGETS = '#img2img_image, #img2img_sketch, #img2maskimg, #inpaint_sketch';

    var SNAP = 8;           // width/height sliders use step=8, and img2img.py already floors to /8
    var HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

    var active = null;      // the canvas currently in crop mode
    var undoStack = {};     // uuid -> {bg, fg} data URLs from before the last crop

    function byId(id) {
        return document.getElementById(id);
    }

    function textarea(uuid, cls) {
        // the selector GradioTextAreaBind itself uses
        return document.querySelector('#' + uuid + '.' + cls + ' textarea');
    }

    function setTextarea(el, value) {
        el.value = value;
        el.dispatchEvent(new Event('input', {bubbles: true}));
    }

    function clamp(v, lo, hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // Everything needed to convert between screen, container and source-image coordinates.
    function geom(uuid) {
        var container = byId('imageContainer_' + uuid);
        var img = byId('image_' + uuid);
        if (!container || !img || !img.complete || !img.naturalWidth) return null;

        var scale = parseFloat(img.style.width) / img.naturalWidth;
        if (!isFinite(scale) || scale <= 0) return null;

        return {
            container: container,
            img: img,
            rect: container.getBoundingClientRect(),
            x: parseFloat(img.style.left) || 0,
            y: parseFloat(img.style.top) || 0,
            scale: scale,
            srcW: img.naturalWidth,
            srcH: img.naturalHeight
        };
    }

    function toSource(g, clientX, clientY) {
        return {
            x: (clientX - g.rect.left - g.x) / g.scale,
            y: (clientY - g.rect.top - g.y) / g.scale
        };
    }

    // ---- toolbar button ------------------------------------------------------------------

    function ensureButtons() {
        var boxes = document.querySelectorAll('.forge-toolbar-box-a');

        for (var i = 0; i < boxes.length; i++) {
            var box = boxes[i];
            if (box.querySelector('.forge-crop-btn')) continue;

            var container = box.closest('.forge-container');
            if (!container || !container.closest(CROP_TARGETS)) continue;

            var center = box.querySelector('[id^="centerButton_"]');
            if (!center) continue;

            var uuid = center.id.replace('centerButton_', '');

            // A canvas that cannot be uploaded to is a read-only output; nothing to crop.
            var upload = byId('uploadButton_' + uuid);
            if (upload && upload.style.display === 'none') continue;

            var crop = document.createElement('button');
            crop.id = 'cropButton_' + uuid;
            crop.className = 'forge-btn forge-no-select forge-crop-btn';
            crop.title = 'Crop';
            crop.textContent = '✂';
            crop.addEventListener('click', function (id) {
                return function (e) {
                    e.stopPropagation();
                    toggleCrop(id);
                };
            }(uuid));

            var revert = document.createElement('button');
            revert.id = 'cropUndoButton_' + uuid;
            revert.className = 'forge-btn forge-no-select forge-crop-undo';
            revert.title = 'Undo crop';
            revert.textContent = '↺';
            revert.style.display = 'none';
            revert.addEventListener('click', function (id) {
                return function (e) {
                    e.stopPropagation();
                    revertCrop(id);
                };
            }(uuid));

            box.appendChild(crop);
            box.appendChild(revert);
        }
    }

    // ---- crop mode -----------------------------------------------------------------------

    function toggleCrop(uuid) {
        if (active && active.uuid === uuid) {
            exitCrop();
        } else {
            exitCrop();
            enterCrop(uuid);
        }
    }

    function enterCrop(uuid) {
        var g = geom(uuid);
        if (!g) return;

        var overlay = document.createElement('div');
        overlay.className = 'forge-crop-overlay';

        var sel = document.createElement('div');
        sel.className = 'forge-crop-sel';
        overlay.appendChild(sel);

        for (var i = 0; i < HANDLES.length; i++) {
            var h = document.createElement('div');
            h.className = 'forge-crop-handle forge-crop-' + HANDLES[i];
            h.setAttribute('data-h', HANDLES[i]);
            sel.appendChild(h);
        }

        var readout = document.createElement('div');
        readout.className = 'forge-crop-size';
        sel.appendChild(readout);

        var bar = document.createElement('div');
        bar.className = 'forge-crop-bar';
        bar.innerHTML =
            '<button class="forge-crop-apply">✓ Apply</button>' +
            '<button class="forge-crop-cancel">✕ Cancel</button>' +
            '<label><input type="checkbox" class="forge-crop-ar"> lock aspect</label>';
        overlay.appendChild(bar);

        g.container.appendChild(overlay);

        active = {
            uuid: uuid,
            overlay: overlay,
            sel: sel,
            readout: readout,
            bar: bar,
            rect: {x: 0, y: 0, w: g.srcW, h: g.srcH},   // source pixels; starts as the whole image
            drag: null
        };

        bar.querySelector('.forge-crop-apply').addEventListener('click', function (e) {
            e.stopPropagation();
            applyCrop();
        });
        bar.querySelector('.forge-crop-cancel').addEventListener('click', function (e) {
            e.stopPropagation();
            exitCrop();
        });
        bar.addEventListener('pointerdown', function (e) {
            e.stopPropagation();
        });

        overlay.addEventListener('pointerdown', onPointerDown);
        // Right-drag pan and wheel zoom deliberately keep bubbling to canvas.min.js.

        var btn = byId('cropButton_' + uuid);
        if (btn) btn.classList.add('forge-crop-active');

        project();
    }

    function exitCrop() {
        if (!active) return;

        var btn = byId('cropButton_' + active.uuid);
        if (btn) btn.classList.remove('forge-crop-active');

        if (active.raf) cancelAnimationFrame(active.raf);
        if (active.overlay.parentNode) active.overlay.parentNode.removeChild(active.overlay);

        active = null;
    }

    // Re-projects the selection every frame, so pan/zoom/maximize need no listeners of their own.
    function project() {
        if (!active) return;

        var g = geom(active.uuid);
        if (g) {
            var r = active.rect;
            active.sel.style.left = (g.x + r.x * g.scale) + 'px';
            active.sel.style.top = (g.y + r.y * g.scale) + 'px';
            active.sel.style.width = (r.w * g.scale) + 'px';
            active.sel.style.height = (r.h * g.scale) + 'px';
            active.readout.textContent = Math.round(r.w) + ' × ' + Math.round(r.h);
        }

        active.raf = requestAnimationFrame(project);
    }

    function aspectLocked() {
        var box = active.bar.querySelector('.forge-crop-ar');
        if (!box || !box.checked) return 0;

        var w = document.querySelector('#img2img_width input[type=number]');
        var h = document.querySelector('#img2img_height input[type=number]');
        if (!w || !h || !parseFloat(h.value)) return 0;

        return parseFloat(w.value) / parseFloat(h.value);
    }

    function onPointerDown(e) {
        if (e.button !== 0) return;     // leave right-drag pan alone

        var g = geom(active.uuid);
        if (!g) return;

        e.stopPropagation();
        e.preventDefault();

        var p = toSource(g, e.clientX, e.clientY);
        var handle = e.target.getAttribute && e.target.getAttribute('data-h');

        if (handle) {
            active.drag = {kind: 'resize', handle: handle};
        } else if (e.target.classList.contains('forge-crop-sel')) {
            active.drag = {kind: 'move', ox: p.x - active.rect.x, oy: p.y - active.rect.y};
        } else {
            active.drag = {kind: 'new', ax: p.x, ay: p.y};
            active.rect = {x: p.x, y: p.y, w: 0, h: 0};
        }

        window.addEventListener('pointermove', onPointerMove, true);
        window.addEventListener('pointerup', onPointerUp, true);
    }

    function onPointerMove(e) {
        if (!active || !active.drag) return;

        var g = geom(active.uuid);
        if (!g) return;

        e.stopPropagation();
        e.preventDefault();

        var p = toSource(g, e.clientX, e.clientY);
        p.x = clamp(p.x, 0, g.srcW);
        p.y = clamp(p.y, 0, g.srcH);

        var r = active.rect;
        var d = active.drag;

        if (d.kind === 'move') {
            r.x = clamp(p.x - d.ox, 0, g.srcW - r.w);
            r.y = clamp(p.y - d.oy, 0, g.srcH - r.h);
        } else {
            var x1, y1, x2, y2;

            if (d.kind === 'new') {
                x1 = Math.min(d.ax, p.x); x2 = Math.max(d.ax, p.x);
                y1 = Math.min(d.ay, p.y); y2 = Math.max(d.ay, p.y);
            } else {
                x1 = r.x; y1 = r.y; x2 = r.x + r.w; y2 = r.y + r.h;
                if (d.handle.indexOf('w') >= 0) x1 = Math.min(p.x, x2);
                if (d.handle.indexOf('e') >= 0) x2 = Math.max(p.x, x1);
                if (d.handle.indexOf('n') >= 0) y1 = Math.min(p.y, y2);
                if (d.handle.indexOf('s') >= 0) y2 = Math.max(p.y, y1);
            }

            r.x = x1; r.y = y1; r.w = x2 - x1; r.h = y2 - y1;

            var ratio = aspectLocked();
            if (ratio) {
                if (r.w / Math.max(r.h, 1) > ratio) {
                    r.w = Math.min(r.h * ratio, g.srcW - r.x);
                    r.h = r.w / ratio;
                } else {
                    r.h = Math.min(r.w / ratio, g.srcH - r.y);
                    r.w = r.h * ratio;
                }
            }
        }
    }

    function onPointerUp(e) {
        if (!active) return;

        e.stopPropagation();
        window.removeEventListener('pointermove', onPointerMove, true);
        window.removeEventListener('pointerup', onPointerUp, true);
        active.drag = null;

        var g = geom(active.uuid);
        if (!g) return;

        // Settle onto whole, SD-friendly pixels.
        var r = active.rect;
        r.x = clamp(Math.round(r.x), 0, g.srcW);
        r.y = clamp(Math.round(r.y), 0, g.srcH);
        r.w = Math.floor(Math.min(Math.round(r.w), g.srcW - r.x) / SNAP) * SNAP;
        r.h = Math.floor(Math.min(Math.round(r.h), g.srcH - r.y) / SNAP) * SNAP;

        if (r.w < SNAP || r.h < SNAP) {
            active.rect = {x: 0, y: 0, w: g.srcW, h: g.srcH};
        }
    }

    // ---- applying ------------------------------------------------------------------------

    function waitFor(test, timeoutMs, done) {
        var waited = 0;
        var timer = setInterval(function () {
            if (test()) {
                clearInterval(timer);
                done(true);
            } else if ((waited += 25) >= timeoutMs) {
                clearInterval(timer);
                done(false);
            }
        }, 25);
    }

    // Pushes a background and a matching foreground, in that order, without racing.
    //
    // canvas.min.js's uploadBase64() onload resizes the drawing canvas whenever the new background
    // has different dimensions -- which clears it -- and its trailing saveState() writes that blank
    // canvas straight back into the foreground textarea. Writing both textareas at once therefore
    // loses the mask, because the two GradioTextAreaBind pollers are independent and both handlers
    // are async. The canvas resize is set synchronously inside that onload, so it is a reliable
    // happens-after marker to gate the foreground push on.
    function pushImages(uuid, bgURL, fgURL, w, h, done) {
        var cv = byId('drawingCanvas_' + uuid);
        var bgTA = textarea(uuid, 'logical_image_background');
        var fgTA = textarea(uuid, 'logical_image_foreground');
        if (!cv || !bgTA || !fgTA) return;

        setTextarea(bgTA, bgURL);

        waitFor(function () {
            return cv.width === w && cv.height === h;
        }, 4000, function (ok) {
            setTextarea(fgTA, fgURL);
            if (!ok) setTimeout(function () { setTextarea(fgTA, fgURL); }, 400);
            if (done) done();
        });
    }

    function applyCrop() {
        var uuid = active.uuid;
        var g = geom(uuid);
        var r = active.rect;

        if (!g) { exitCrop(); return; }

        var x = Math.round(r.x), y = Math.round(r.y), w = Math.round(r.w), h = Math.round(r.h);
        if (w <= 0 || h <= 0 || (x === 0 && y === 0 && w === g.srcW && h === g.srcH)) {
            exitCrop();
            return;
        }

        var cv = byId('drawingCanvas_' + uuid);

        var bg = document.createElement('canvas');
        bg.width = w; bg.height = h;
        bg.getContext('2d').drawImage(g.img, x, y, w, h, 0, 0, w, h);

        var fg = document.createElement('canvas');
        fg.width = w; fg.height = h;
        if (cv) fg.getContext('2d').drawImage(cv, x, y, w, h, 0, 0, w, h);

        var bgURL, fgURL;
        try {
            bgURL = bg.toDataURL('image/png');
            fgURL = fg.toDataURL('image/png');
        } catch (err) {
            console.warn('forge crop: could not read the canvas', err);
            exitCrop();
            return;
        }

        var bgTA = textarea(uuid, 'logical_image_background');
        var fgTA = textarea(uuid, 'logical_image_foreground');
        undoStack[uuid] = {bg: bgTA.value, fg: fgTA.value, w: g.srcW, h: g.srcH};

        var undoBtn = byId('cropUndoButton_' + uuid);
        if (undoBtn) undoBtn.style.display = '';

        exitCrop();
        pushImages(uuid, bgURL, fgURL, w, h);
    }

    // canvas.min.js's own undo history only holds the scribble layer -- the background is never in
    // it -- so its undo button cannot bring a crop back, and after one it would putImageData a
    // now-oversized mask that silently clips. Hence our own one-step revert.
    function revertCrop(uuid) {
        var saved = undoStack[uuid];
        if (!saved) return;

        exitCrop();
        pushImages(uuid, saved.bg, saved.fg, saved.w, saved.h, function () {
            delete undoStack[uuid];
            var btn = byId('cropUndoButton_' + uuid);
            if (btn) btn.style.display = 'none';
        });
    }

    // ---- bootstrap -----------------------------------------------------------------------

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && active) {
            e.stopPropagation();
            exitCrop();
        }
    }, true);

    var queued = false;
    function schedule() {
        if (queued) return;
        queued = true;
        requestAnimationFrame(function () {
            queued = false;
            ensureButtons();
        });
    }

    if (typeof MutationObserver === 'function') {
        new MutationObserver(schedule).observe(document.documentElement, {childList: true, subtree: true});
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', schedule);
    } else {
        schedule();
    }
})();
