/* Uploaded photos stay in this page until processed or removed. */
const peopleUI = { files: [], urls: [], busy: false, mode: "enrol", person: null };
const photoMessages = {
  accepted: "✓ Face accepted", no_face: "No face detected", multiple_faces: "Multiple faces detected",
  face_too_small: "Face is too small", blurred: "Face is too blurry", clipped_at_edge: "Face is cropped",
  low_confidence: "Face could not be detected confidently", too_dark: "Photo is too dark",
  too_bright: "Photo is too bright", extreme_pose: "Face angle is too extreme", invalid_crop: "Face is cropped",
  unusable: "Photo did not pass quality checks", invalid_image: "Invalid image", embed_failed: "Could not create face sample",
};

function clearPeoplePhotos() {
  peopleUI.urls.forEach(URL.revokeObjectURL);
  peopleUI.urls = [];
  peopleUI.files = [];
}

function photoStatusMessage(result) {
  if (!result) return "Waiting for validation";
  return result.message || (result.reasons?.length
    ? result.reasons.map(reason => photoMessages[reason] || "Quality check failed").join("; ")
    : photoMessages[result.status] || "Quality check failed");
}

async function renderPeople() {
  clearPeoplePhotos();
  view.innerHTML = `<section class="people">${pageHeading("IDENTITY DIRECTORY", "People", "Known-person profiles and presentation-ready face thumbnails.", '<span class="quiet-badge">Live recognition off</span>')}
    <div class="people-layout"><section class="detail"><h3 id="people-form-title">Remember someone new</h3>
    <form id="people-form"><fieldset id="people-fields">
    <div id="people-profile"><label><span>Name <span aria-hidden="true">*</span></span><input name="name" required maxlength="120" autocomplete="off"></label>
    <label>Relationship / context<input name="relationship" maxlength="4000" placeholder="e.g. Mum, friend, neighbour"></label>
    <label>Hobbies<input name="hobbies" maxlength="4000" placeholder="Gardening, cooking, travel"></label>
    <label>Interests<input name="interests" maxlength="4000" placeholder="Jazz, Japanese food"></label>
    <label>Notes / details<textarea name="notes" rows="3" maxlength="4000"></textarea></label></div>
    <div id="people-upload"><label class="photo-drop" id="photo-drop">Choose photos or drop them here
    <input id="people-photos" type="file" accept="image/jpeg,image/png" multiple></label>
    <p class="form-help">Upload several clear photos of the same person. Different angles, lighting and appearance can improve recognition. Recommended: 3–10 photos; one is enough to start.</p>
    <details><summary>Tips for good photos</summary><p>One person per photo, face clearly visible. Avoid heavy blur and severe cropping. Try slightly different angles and lighting, and glasses/no-glasses variation.</p></details>
    <p class="form-help">JPG / PNG · 8 MiB each · 32 MiB total · up to 16 megapixels. Originals are processed temporarily and never saved.</p>
    <div id="photo-list" class="photo-list"></div></div>
    <div class="actions"><button class="btn btn--primary" id="people-save">Remember person</button>
    <button type="button" class="btn" id="people-cancel">Clear</button></div></fieldset></form>
    <p id="people-message" role="status" aria-live="polite"></p></section>
    <section><div class="section-heading"><div><h3>Known people</h3><p>Only persisted identity records are shown.</p></div><button id="people-refresh" class="btn">Refresh</button></div>
    <div id="people-list" class="people-list"><p>Loading people…</p></div></section></div></section>`;
  peopleUI.mode = "enrol";
  peopleUI.person = null;
  const picker = document.getElementById("people-photos");
  picker.onchange = () => { selectPeoplePhotos(picker.files); picker.value = ""; };
  const drop = document.getElementById("photo-drop");
  drop.ondragover = (event) => { event.preventDefault(); };
  drop.ondrop = (event) => { event.preventDefault(); if (!peopleUI.busy) selectPeoplePhotos(event.dataTransfer.files); };
  document.getElementById("people-cancel").onclick = resetPeopleForm;
  document.getElementById("people-form").onsubmit = savePerson;
  document.getElementById("people-refresh").onclick = loadPeople;
  await loadPeople();
}

function peopleMessage(message) {
  const element = document.getElementById("people-message");
  if (element) element.textContent = message;
}

function encodedPhotoBytes(fileSize) {
  return Math.ceil(fileSize / 3) * 4;
}

function selectPeoplePhotos(files) {
  const selected = [...peopleUI.files, ...files];
  const binaryTotal = selected.reduce((sum, file) => sum + file.size, 0);
  // JSON embeds base64; reject before upload if the encoded body would exceed the server limit.
  const estimatedBody = selected.reduce((sum, file) => sum + encodedPhotoBytes(file.size), 0) + 65536;
  if (selected.length > 10 || selected.some(file => file.size > 8 * 1024 * 1024) ||
      binaryTotal > 32 * 1024 * 1024 || estimatedBody > 48 * 1024 * 1024) {
    peopleMessage("Choose up to 10 photos, 8 MiB each and 32 MiB combined (encoded upload limit 48 MiB)."); return;
  }
  if (selected.some(file => !["image/jpeg", "image/png"].includes(file.type))) {
    peopleMessage("Choose JPG or PNG photos."); return;
  }
  peopleUI.files = selected;
  drawPeoplePhotos();
}

function drawPeoplePhotos(results = []) {
  peopleUI.urls.forEach(URL.revokeObjectURL);
  peopleUI.urls = peopleUI.files.map(file => URL.createObjectURL(file));
  const list = document.getElementById("photo-list");
  if (!list) return;
  list.innerHTML = peopleUI.files.map((file, index) => {
    const result = results.find(item => item.index === index);
    const message = photoStatusMessage(result);
    return `<article class="photo-item"><img src="${peopleUI.urls[index]}" alt="Selected photo ${index + 1}">
      <div><strong>${escapeHtml(file.name)}</strong><p>${escapeHtml(message)}</p></div>
      <button type="button" class="btn" data-remove-photo="${index}" aria-label="Remove photo ${index + 1}" ${peopleUI.busy ? "disabled" : ""}>×</button></article>`;
  }).join("");
  list.querySelectorAll("[data-remove-photo]").forEach(button => button.onclick = () => {
    peopleUI.files.splice(Number(button.dataset.removePhoto), 1); drawPeoplePhotos();
  });
}

function resetPeopleForm() {
  clearPeoplePhotos();
  peopleUI.mode = "enrol"; peopleUI.person = null;
  document.getElementById("people-form").reset();
  document.getElementById("people-profile").hidden = false;
  document.querySelector('[name="name"]').required = true;
  document.getElementById("people-upload").hidden = false;
  document.getElementById("people-form-title").textContent = "Remember someone new";
  document.getElementById("people-save").textContent = "Remember person";
  document.getElementById("people-cancel").textContent = "Clear";
  drawPeoplePhotos(); peopleMessage("");
}

async function loadPeople() {
  const list = document.getElementById("people-list");
  if (!list) return;
  try {
    const response = await api("/api/people");
    if (response.error) throw new Error(response.error);
    list.innerHTML = response.people.length ? "" : '<div class="empty-state"><h3>A familiar face starts here</h3><p>Add someone using the form. Recognition and automatic greetings stay off.</p></div>';
    response.people.forEach(person => {
      const card = document.createElement("article"); card.className = "person-card";
      const thumbnail = person.avatar_url || person.thumbnail_url || person.face_crop_url;
      const avatar = thumbnail
        ? `<div class="person-avatar"><img src="${escapeHtml(thumbnail)}" alt="${escapeHtml(person.name)} face thumbnail" loading="lazy"></div>`
        : `<div class="person-avatar" aria-hidden="true">${escapeHtml(person.name.slice(0, 1))}</div>`;
      card.innerHTML = `${avatar}
        <div><h3>${escapeHtml(person.name)}</h3><p>${escapeHtml(person.relationship || "Known person")}</p>
        <span class="person-state"><span class="status-dot status-dot--${person.embedding_count > 0 ? "online" : "unknown"}"></span>${person.embedding_count > 0 ? "Enrolled" : "Not enrolled"}</span>
        <p>Last seen: ${escapeHtml(person.last_seen || "Unavailable")}</p>
        <span class="quiet-badge">${person.embedding_count} persisted face sample${person.embedding_count === 1 ? "" : "s"}</span>
        <p>${escapeHtml([...person.hobbies, ...person.interests].join(" · "))}</p>
        <p class="person-notes">${escapeHtml(person.notes.join("\n"))}</p>
        <div class="actions"><button class="btn" data-mode="edit">Edit details</button><button class="btn" data-mode="add">Add photos</button>
        <button class="btn btn--danger" data-mode="forget">Forget</button></div></div>`;
      card.querySelectorAll("[data-mode]").forEach(button => button.onclick = async () => {
        if (peopleUI.busy) return;
        const mode = button.dataset.mode;
        if (mode === "forget") {
          if (!confirm(`Forget ${person.name}?\nThis will remove their face identity and saved profile.`)) return;
          peopleUI.busy = true; button.disabled = true;
          try {
            const result = await api("/api/people/forget", { method: "POST", body: JSON.stringify({person_id: person.person_id, confirmed: true}) });
            if (result.persisted !== true || result.status !== "forgotten") throw new Error("Deletion could not be verified.");
            resetPeopleForm(); peopleMessage(`${person.name} has been forgotten.`); await loadPeople();
          } catch (error) { peopleMessage(error.message); }
          finally { peopleUI.busy = false; button.disabled = false; }
          return;
        }
        resetPeopleForm(); peopleUI.mode = mode; peopleUI.person = person;
        const form = document.getElementById("people-form");
        for (const key of ["name", "relationship", "hobbies", "interests", "notes"]) {
          form.elements[key].value = Array.isArray(person[key]) ? person[key].join(key === "notes" ? "\n" : ", ") : person[key] || "";
        }
        document.getElementById("people-profile").hidden = mode === "add";
        form.elements.name.required = mode !== "add";
        document.getElementById("people-upload").hidden = mode === "edit";
        document.getElementById("people-form-title").textContent = `${mode === "edit" ? "Edit details" : "Add photos"} · ${person.name}`;
        document.getElementById("people-save").textContent = mode === "edit" ? "Save details" : "Add photos";
        document.getElementById("people-cancel").textContent = "Cancel";
        form.scrollIntoView({ block: "nearest" });
      });
      list.append(card);
    });
  } catch (error) { list.textContent = `Could not load people: ${error.message}`; }
}

async function savePerson(event) {
  event.preventDefault();
  if (peopleUI.busy) return;
  const form = event.currentTarget;
  if (peopleUI.mode !== "edit" && !peopleUI.files.length) { peopleMessage("Choose at least one photo."); return; }
  const body = Object.fromEntries(new FormData(form));
  body.person_id = peopleUI.person?.person_id;
  peopleUI.busy = true; document.getElementById("people-fields").disabled = true;
  peopleMessage("Preparing upload…");
  try {
    body.photos = [];
    for (const file of peopleUI.files) {
      const content = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = () => reject(new Error("Could not read photo")); reader.readAsDataURL(file);
      });
      body.photos.push({ type: file.type, content });
    }
    peopleMessage(peopleUI.mode === "edit" ? "Saving and verifying details…" : "Uploading, validating photos, generating face samples and verifying saved memory…");
    let response;
    try {
      response = await fetch(`/api/people/${peopleUI.mode}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
    } catch (error) {
      const detail = error && error.message ? error.message : "network error";
      throw new Error(
        detail === "Failed to fetch"
          ? "Could not reach the People API. Restart the local dashboard (routes must include /api/people) and retry."
          : detail,
      );
    }
    let result = {};
    try {
      result = await response.json();
    } catch (_error) {
      throw new Error(
        response.status === 404
          ? "People API is missing on this dashboard process. Restart the local dashboard and retry."
          : `Dashboard returned an unreadable response (HTTP ${response.status}).`,
      );
    }
    drawPeoplePhotos(result.photos || []);
    if (!response.ok || result.persisted !== true || !result.person_id || result.embedding_count < 1) {
      throw new Error(result.error || result.message || "Saved memory could not be verified.");
    }
    const accepted = (result.photos || []).filter(photo => photo.status === "accepted").length;
    const message = peopleUI.mode === "enrol" ? `${result.name} has been added to Reachy's memory.` : `${result.name}'s memory has been updated.`;
    const validationSummary = (result.photos || []).map((photo, index) => `${peopleUI.files[index]?.name || `Photo ${index + 1}`}: ${photoStatusMessage(photo)}`).join("\n");
    resetPeopleForm();
    peopleMessage(`${message}${accepted ? ` ${accepted} of ${result.photos.length} photos accepted.` : ""}${validationSummary ? "\n" + validationSummary : ""}`);
    await loadPeople();
  } catch (error) { peopleMessage(error.message); }
  finally {
    peopleUI.busy = false;
    const fields = document.getElementById("people-fields"); if (fields) fields.disabled = false;
    document.querySelectorAll("[data-remove-photo]").forEach(button => { button.disabled = false; });
  }
}
