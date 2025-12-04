// Progress bar with queue tracking and reconnection support

function rememberGallerySelection() {}
function getGallerySelectedIndex() {}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}


var originalAppTitle = undefined;

onUiLoaded(function() {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    var title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}


function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

// Progress request with reconnection support
// inactivityTimeout: time in ms before giving up on a task that was never queued/active
// Setting to 0 disables the timeout (for restore operations)
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 60000) {
    var dateStart = new Date();
    var wasEverActive = false;
    var wasEverQueued = false;
    var parentProgressbar = progressbarContainer.parentNode;
    var wakeLock = null;
    var isDestroyed = false;
    var reconnectAttempts = 0;
    var maxReconnectAttempts = 10;

    var requestWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock !== null) return;
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.error('Wake Lock is not supported.');
            wakeLock = false;
        }
    };

    var releaseWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error('Wake Lock release failed', err);
        }
    };

    var divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    var divInner = document.createElement('div');
    divInner.className = 'progress';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    var livePreview = null;

    var removeProgressBar = function() {
        isDestroyed = true;
        releaseWakeLock();
        if (!divProgress) return;

        setTitle("");
        try {
            parentProgressbar.removeChild(divProgress);
        } catch (e) {}
        if (gallery && livePreview) {
            try {
                gallery.removeChild(livePreview);
            } catch (e) {}
        }
        atEnd();

        divProgress = null;
    };

    var handleError = function() {
        reconnectAttempts++;
        if (reconnectAttempts >= maxReconnectAttempts) {
            console.error('Max reconnect attempts reached, giving up');
            removeProgressBar();
            return;
        }
        // Exponential backoff: 1s, 2s, 3s, 4s, 5s (capped)
        var delay = Math.min(reconnectAttempts, 5) * 1000;
        console.log('Connection error, reconnecting in', delay, 'ms (attempt', reconnectAttempts + ')');
        setTimeout(function() {
            if (!isDestroyed) {
                funProgress(id_task);
            }
        }, delay);
    };

    var funProgress = function(id_task) {
        if (isDestroyed) return;

        requestWakeLock();
        request("./internal/progress", {id_task: id_task, live_preview: false}, function(res) {
            if (isDestroyed) return;

            // Reset reconnect counter on success
            reconnectAttempts = 0;

            if (res.completed) {
                removeProgressBar();
                return;
            }

            let progressText = "";

            divInner.style.width = ((res.progress || 0) * 100.0) + '%';
            divInner.style.background = res.progress ? "" : "transparent";

            if (res.progress > 0) {
                progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%';
            }

            if (res.eta) {
                progressText += " ETA: " + formatTime(res.eta);
            }

            setTitle(progressText);

            if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
                progressText = res.textinfo + " " + progressText;
            }

            divInner.textContent = progressText;

            var elapsedFromStart = (new Date() - dateStart) / 1000;

            // Track if we've ever been in the queue or active
            if (res.active) {
                wasEverActive = true;
                wasEverQueued = true;
            }
            if (res.queued) {
                wasEverQueued = true;
            }

            // Task finished (was active, now isn't)
            if (!res.active && wasEverActive) {
                removeProgressBar();
                return;
            }

            // Timeout check - only if we've never been queued/active and timeout is enabled
            if (inactivityTimeout > 0 && elapsedFromStart * 1000 > inactivityTimeout && !res.queued && !res.active && !wasEverQueued) {
                console.log('Inactivity timeout reached for task', id_task);
                removeProgressBar();
                return;
            }

            if (onProgress) {
                onProgress(res);
            }

            setTimeout(() => {
                funProgress(id_task);
            }, opts.live_preview_refresh_period || 500);
        }, handleError);
    };

    var funLivePreview = function(id_task, id_live_preview) {
        if (isDestroyed) return;

        request("./internal/progress", {id_task: id_task, id_live_preview: id_live_preview, live_preview: true}, function(res) {
            if (!divProgress || isDestroyed) {
                return;
            }

            if (res.live_preview && gallery) {
                var img = new Image();
                img.onload = function() {
                    if (isDestroyed) return;
                    if (!livePreview) {
                        livePreview = document.createElement('div');
                        livePreview.className = 'livePreview';
                        gallery.insertBefore(livePreview, gallery.firstElementChild);
                    }

                    livePreview.appendChild(img);
                    if (livePreview.childElementCount > 2) {
                        livePreview.removeChild(livePreview.firstElementChild);
                    }
                };
                img.src = res.live_preview;
            }

            setTimeout(() => {
                funLivePreview(id_task, res.id_live_preview || id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            // Don't give up on live preview errors, just retry
            if (!isDestroyed) {
                setTimeout(() => {
                    funLivePreview(id_task, id_live_preview);
                }, 2000);
            }
        });
    };

    funProgress(id_task);

    if (gallery) {
        funLivePreview(id_task, 0);
    }
}
