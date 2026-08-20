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
})();
