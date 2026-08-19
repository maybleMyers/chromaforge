// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    var visibleGalleryButtons = [];
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
}

function gallery_container_buttons(gallery_container) {
    return gradioApp().querySelectorAll(`#${gallery_container} .thumbnail-item.thumbnail-small`);
}

function selected_gallery_index_id(gallery_container) {
    return Array.from(gallery_container_buttons(gallery_container)).findIndex(elem => elem.classList.contains('selected'));
}

function extract_image_from_gallery(gallery) {
    if (gallery.length == 0) {
        return [null];
    }

    var index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // Use the first image in the gallery as the default
        index = 0;
    }

    return [[gallery[index]]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}

// Tabname-aware variants of the img2img helpers below. The img2img generation panel is
// built once per tab (Img2img, LLM2img), so anything that reaches into the DOM has to be
// told which "mode_<tabname>" Tabs group it is looking at.
function switchToModeTab(modeId, no) {
    var mode = gradioApp().getElementById(modeId);
    if (mode) mode.querySelectorAll('button')[no].click();
    return [];
}

// Select a top-level tab by its interface id. Deliberately not a hardcoded nav index the
// way switch_to_txt2img/switch_to_img2img are: a tab's position depends on
// opts.ui_tab_order and opts.hidden_tabs. The nav buttons are in the same order as the tab
// content divs, which modules/ui.py gives elem_id="tab_<ifid>".
function switchToTopTab(ifid) {
    var tabs = gradioApp().querySelector('#tabs');
    if (!tabs) return;
    var nav = tabs.querySelector(':scope > div.tab-nav');
    if (!nav) return;
    var contents = Array.from(tabs.children).filter(function(el) {
        return el.id && el.id.indexOf('tab_') === 0;
    });
    var idx = contents.findIndex(function(el) {
        return el.id === 'tab_' + ifid;
    });
    if (idx >= 0 && nav.children[idx]) nav.children[idx].click();
}

// Destinations for the LLM2img gallery's send-to buttons. connect_paste_params_buttons()
// in modules/infotext_utils.py attaches these by name as switch_to_<destination>, so the
// names must match the paste destinations registered in modules/ui_img2img_panel.py.
// Mode indices come from MODE_TAB_INDEX there: img2img 0, inpaint 3.
function switch_to_llm2img() {
    switchToTopTab('llm2img');
    switchToModeTab('mode_llm2img', 0);
    return Array.from(arguments);
}

function switch_to_llm2img_inpaint() {
    switchToTopTab('llm2img');
    switchToModeTab('mode_llm2img', 3);
    return Array.from(arguments);
}

function currentModeSourceResolution(modeId, w, h, r) {
    var img = gradioApp().querySelector('#' + modeId + ' > div[style="display: block;"] :is(img, canvas)');
    return img ? [img.naturalWidth || img.width, img.naturalHeight || img.height, r] : [0, 0, r];
}

function currentModeExpandResolution(modeId) {
    var args = Array.from(arguments).slice(1);
    var img = gradioApp().querySelector('#' + modeId + ' > div[style="display: block;"] :is(img, canvas)');
    args[0] = img ? (img.naturalWidth || img.width) : 0;
    args[1] = img ? (img.naturalHeight || img.height) : 0;
    return args;
}

function getModeTabIndexArgs(modeId) {
    let res = Array.from(arguments).slice(1);
    res.splice(-2);
    res[0] = get_tab_index(modeId);
    return res;
}

// Where a tab wants its live preview drawn. Tabs absent from this map get it over their
// gallery, as upstream does; see modules/ui_common.py for why LLM2img does not.
var livePreviewHost = {
    llm2img: 'llm2img_live_preview',
};

function livePreviewElementFor(tabname) {
    return gradioApp().getElementById(livePreviewHost[tabname] || (tabname + '_gallery'));
}

function restoreProgressForTab(tabname) {
    showRestoreProgressButton(tabname, false);

    var id = localGet(tabname + "_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder(tabname);
        requestProgress(id, gradioApp().getElementById(tabname + '_gallery_container'), livePreviewElementFor(tabname), function() {
            showSubmitButtons(tabname, true);
        }, null, 0);
    }

    return id;
}
function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_ref_edit() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(4);
    return Array.from(arguments);
}

function switch_to_extras() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[3].click();

    return Array.from(arguments);
}

function get_tab_index(tabId) {
    let buttons = gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].classList.contains('selected')) {
            return i;
        }
    }
    return 0;
}

function create_tab_index_args(tabId, args) {
    var res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    let res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index('mode_img2img');
    return res;
}

function create_submit_args(args) {
    var res = Array.from(args);

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.

    // If gradio at some point stops sending outputs, this may break something
    if (Array.isArray(res[res.length - 4])) {
        //res[res.length - 4] = null;
        // simply drop output args
        res = res.slice(0, res.length - 4);
    } else if (Array.isArray(res[res.length - 3])) {
        // for submit_extras()
        //res[res.length - 3] = null;
        res = res.slice(0, res.length - 3);
    }

    return res;
}

function setSubmitButtonsVisibility(tabname, showInterrupt, showSkip, showInterrupting) {
    gradioApp().getElementById(tabname + '_interrupt').style.display = showInterrupt ? "block" : "none";
    gradioApp().getElementById(tabname + '_skip').style.display = showSkip ? "block" : "none";
    gradioApp().getElementById(tabname + '_interrupting').style.display = showInterrupting ? "block" : "none";
}

function showSubmitButtons(tabname, show) {
    setSubmitButtonsVisibility(tabname, !show, !show, false);
}

function showSubmitInterruptingPlaceholder(tabname) {
    setSubmitButtonsVisibility(tabname, false, true, true);
}

function showRestoreProgressButton(tabname, show) {
    var button = gradioApp().getElementById(tabname + "_restore_progress");
    if (!button) return;
    button.style.setProperty('display', show ? 'flex' : 'none', 'important');
}

function submit() {
    showSubmitButtons('txt2img', false);

    var id = randomId();
    localSet("txt2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        showSubmitButtons('txt2img', true);
        localRemove("txt2img_task_id");
        showRestoreProgressButton('txt2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function submit_txt2img_upscale() {
    var res = submit(...arguments);

    res[2] = selected_gallery_index();

    return res;
}

function submit_img2img() {
    showSubmitButtons('img2img', false);

    var id = randomId();
    localSet("img2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
        showSubmitButtons('img2img', true);
        localRemove("img2img_task_id");
        showRestoreProgressButton('img2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function submit_llm2img() {
    showSubmitButtons('llm2img', false);

    var id = randomId();
    localSet("llm2img_task_id", id);

    // The live preview goes into its own pane, not the gallery: this task spans the whole
    // multi-round loop, and the .livePreview overlay is only torn down when the task ends,
    // so over the gallery it would hide every round's results until the run finished.
    requestProgress(id, gradioApp().getElementById('llm2img_gallery_container'), livePreviewElementFor('llm2img'), function() {
        showSubmitButtons('llm2img', true);
        localRemove("llm2img_task_id");
        showRestoreProgressButton('llm2img', false);
    });

    // Same trick as create_submit_args(), but this event has six outputs rather than four,
    // so the echoed gallery sits six from the end instead of four.
    var res = Array.from(arguments);
    if (Array.isArray(res[res.length - 6])) {
        res = res.slice(0, res.length - 6);
    }

    res[0] = id;

    return res;
}

function submit_extras() {
    showSubmitButtons('extras', false);

    var id = randomId();

    requestProgress(id, gradioApp().getElementById('extras_gallery_container'), gradioApp().getElementById('extras_gallery'), function() {
        showSubmitButtons('extras', true);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function restoreProgressTxt2img() {
    showRestoreProgressButton("txt2img", false);
    var id = localGet("txt2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder('txt2img');
        requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
            showSubmitButtons('txt2img', true);
        }, null, 0);
    }

    return id;
}

function restoreProgressImg2img() {
    showRestoreProgressButton("img2img", false);

    var id = localGet("img2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder('img2img');
        requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
            showSubmitButtons('img2img', true);
        }, null, 0);
    }

    return id;
}


/**
 * Configure the width and height elements on `tabname` to accept
 * pasting of resolutions in the form of "width x height".
 */
function setupResolutionPasting(tabname) {
    var width = gradioApp().querySelector(`#${tabname}_width input[type=number]`);
    var height = gradioApp().querySelector(`#${tabname}_height input[type=number]`);
    for (const el of [width, height]) {
        el.addEventListener('paste', function(event) {
            var pasteData = event.clipboardData.getData('text/plain');
            var parsed = pasteData.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (parsed) {
                width.value = parsed[1];
                height.value = parsed[2];
                updateInput(width);
                updateInput(height);
                event.preventDefault();
            }
        });
    }
}

onUiLoaded(function() {
    showRestoreProgressButton('txt2img', localGet("txt2img_task_id"));
    showRestoreProgressButton('img2img', localGet("img2img_task_id"));
    setupResolutionPasting('txt2img');
    setupResolutionPasting('img2img');
});


function modelmerger() {
    var id = randomId();
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function() {});

    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    var name_ = prompt('Style name:');
    return [name_, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}


var opts = {};
onAfterUiUpdate(function() {
    if (Object.keys(opts).length != 0) return;

    var json_elem = gradioApp().getElementById('settings_json');
    if (json_elem == null) return;

    var textarea = json_elem.querySelector('textarea');
    var jsdata = textarea.value;
    opts = JSON.parse(jsdata);

    executeCallbacks(optionsAvailableCallbacks); /*global optionsAvailableCallbacks*/
    executeCallbacks(optionsChangedCallbacks); /*global optionsChangedCallbacks*/

    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display = "none";
});

onOptionsChanged(function() {
    var elem = gradioApp().getElementById('sd_checkpoint_hash');
    var sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    var shorthash = sd_checkpoint_hash.substring(0, 10);

    if (elem && elem.textContent != shorthash) {
        elem.textContent = shorthash;
        elem.title = sd_checkpoint_hash;
        elem.href = "https://google.com/search?q=" + sd_checkpoint_hash;
    }
});

let txt2img_textarea, img2img_textarea = undefined;

function restart_reload() {
    document.body.style.backgroundColor = "var(--background-fill-primary)";
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    var requestPing = function() {
        requestGet("./internal/ping", {}, function(data) {
            location.reload();
        }, function() {
            setTimeout(requestPing, 500);
        });
    };

    setTimeout(requestPing, 2000);

    return [];
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target) {
    let e = new Event("input", {bubbles: true});
    Object.defineProperty(e, "target", {value: target});
    target.dispatchEvent(e);
}


var desiredCheckpointName = null;
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    gradioApp().getElementById('change_checkpoint').click();
}
var desiredVAEName = 0;
function selectVAE(vae) {
    desiredVAEName = vae;
}

function currentImg2imgExpandResolution() {
    var args = Array.from(arguments);
    var img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] :is(img, canvas)');
    args[0] = img ? (img.naturalWidth || img.width) : 0;
    args[1] = img ? (img.naturalHeight || img.height) : 0;
    return args;
}

function currentImg2imgSourceResolution(w, h, r) {
    var img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] :is(img, canvas)');
    return img ? [img.naturalWidth || img.width, img.naturalHeight || img.height, r] : [0, 0, r];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    // At the time this is called from gradio, the image has no yet been replaced.
    // There may be a better solution, but this is simple and straightforward so I'm going with it.

    setTimeout(function() {
        gradioApp().getElementById('img2img_update_resize_to').click();
    }, 500);

    return [];

}



function setRandomSeed(elem_id) {
    var input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}

function switchWidthHeight(tabname) {
    var width = gradioApp().querySelector("#" + tabname + "_width input[type=number]");
    var height = gradioApp().querySelector("#" + tabname + "_height input[type=number]");
    if (!width || !height) return [];

    var tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    updateInput(width);
    updateInput(height);
    return [];
}


var onEditTimers = {};

// calls func after afterMs milliseconds has passed since the input elem has been edited by user
function onEdit(editId, elem, afterMs, func) {
    var edited = function() {
        var existingTimer = onEditTimers[editId];
        if (existingTimer) clearTimeout(existingTimer);

        onEditTimers[editId] = setTimeout(func, afterMs);
    };

    elem.addEventListener("input", edited);

    return edited;
}
