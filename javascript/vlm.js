// Page-level plumbing for the VLM tab (modules/ui_vlm.py rendering vlm.py's interface).
//
// Standalone, vlm.py registers the same code through demo.load(js=...). Embedded it cannot:
// Gradio 4 drops head= on a child Blocks, and a <script> tag inserted through gr.HTML never
// executes. This file is picked up by modules/ui_gradio_extensions.py:javascript_html() and
// lands in the real document head, so keep it in step with VLM_PAGE_JS in vlm.py.

(function() {
    // A file dropped outside an upload zone would otherwise navigate the browser to the
    // file, wiping out the page. Swallow stray drops, but only inside the VLM tab: a
    // window-wide handler would also sit under the img2img and ControlNet canvases, which
    // do their own drop handling.
    function guardStrayDrop(e) {
        if (e.target && e.target.closest && e.target.closest('#tab_vlm')) {
            e.preventDefault();
        }
    }

    window.addEventListener('dragover', guardStrayDrop, false);
    window.addEventListener('drop', guardStrayDrop, false);

    // Per-card remove buttons in the media preview: write the 1-based index into the hidden
    // textbox, then click the hidden button.
    window.vlmRemoveMedia = (idx) => {
        const box = document.querySelector('#vlm_media_remove_idx textarea, #vlm_media_remove_idx input');
        if (!box) return;
        box.value = String(idx);
        box.dispatchEvent(new Event('input', { bubbles: true }));
        setTimeout(() => {
            const btn = document.querySelector('#vlm_media_remove_btn button, button#vlm_media_remove_btn, #vlm_media_remove_btn');
            if (btn) btn.click();
        }, 60);
    };

    // Streaming text arrives as deltas - each frame carries only the characters that turned
    // up since the last one - and is appended as a text node. Handing the whole reply to a
    // Chatbot or Markdown component instead means re-serialising and re-parsing all of it on
    // every frame, which costs O(reply) per frame and is quadratic over a run; at 70 tok/s
    // that is more than a second of rendering per second of generation, and because Gradio
    // suspends the generator for the whole round trip, nothing drains the socket while it
    // happens and the backpressure stalls llama-server itself.
    //
    // Appending text nodes has a second benefit here: hints.js only inspects ELEMENT_NODEs,
    // so its tooltip scan - which would otherwise walk the whole growing reply on every
    // frame, under the app-wide observer in script.js - never sees any of this.
    window.vlmPipeDelta = (srcId, dstId) => {
        const src = document.querySelector('#' + srcId);
        const dst = document.querySelector('#' + dstId);
        if (!src || !dst) return false;
        if (src.dataset.vlmPiped) return true;
        src.dataset.vlmPiped = '1';
        dst.textContent = '';   // take the element off Gradio and own its contents
        new MutationObserver(() => {
            const span = src.querySelector('span[data-seq]');
            // data-seq changes every frame, so an identical delta twice running still
            // registers as a change rather than being silently dropped.
            if (!span || span.dataset.seq === src.dataset.lastSeq) return;
            src.dataset.lastSeq = span.dataset.seq;
            if (span.dataset.reset) dst.textContent = span.textContent;
            else dst.appendChild(document.createTextNode(span.textContent));
            dst.scrollTop = dst.scrollHeight;
        }).observe(src, { childList: true, subtree: true, characterData: true });
        return true;
    };

    // The tab is built lazily, so the carriers may not exist yet. onUiLoaded is Forge's and
    // does not exist standalone, where vlm.py runs this same code through demo.load - so
    // poll instead of depending on either.
    (function bootstrap(tries) {
        const think = window.vlmPipeDelta('vlm_delta_think', 'vlm_live_think');
        const answer = window.vlmPipeDelta('vlm_delta_answer', 'vlm_live_answer');
        if (think && answer) return;
        if (tries > 0) setTimeout(() => bootstrap(tries - 1), 250);
    })(120);
})();
