let dismissTimer;

// Popovers stay above native modal dialogs; keep the live region in the active dialog.
function notifyCopied(content) {
  const snackbar = document.getElementById('snackbar');
  clearTimeout(dismissTimer);
  if (snackbar.matches(':popover-open')) snackbar.hidePopover();
  const host = document.activeElement?.closest('dialog[open]')
    || [...document.querySelectorAll('dialog[open]')].at(-1) || document.body;
  if (snackbar.parentElement !== host) host.append(snackbar);
  snackbar.textContent = `${content} copied to clipboard`;
  snackbar.showPopover();
  dismissTimer = setTimeout(() => {
    if (snackbar.matches(':popover-open')) snackbar.hidePopover();
  }, 3200);
}

export async function copyToClipboard(text, content) {
  if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
  await navigator.clipboard.writeText(text);
  notifyCopied(content);
}
