document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("visual-hash-canvas");
  const inputBox = document.getElementById("name");
  const randomNameButton = document.getElementById("random-name-btn");

  if (canvas) {
    const rotationSpeed = 0.5;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.src = canvas.getAttribute("data-image");

    let rotation = 0;
    img.onload = function () {
      animate();
    };

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate((rotation * Math.PI) / 180);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);
      ctx.restore();

      rotation += rotationSpeed;
      rotation %= 360;
      requestAnimationFrame(animate);
    }

    const downloadBtnContainer = document.getElementById(
      "download-btn-container"
    );
    const downloadBtn = document.createElement("form");

    downloadBtn.action = "/download-plot";
    downloadBtn.method = "post";
    downloadBtn.innerHTML = `
        <input type="hidden" name="name" value="${canvas.getAttribute(
          "data-name"
        )}" />
        <button type="submit" style="margin-left:0.25rem">Download</button>
      `;
    downloadBtnContainer.appendChild(downloadBtn);

    const generateBtn = document.querySelector(".generate");
    generateBtn.style.marginRight = "0.25rem";

    const randomNameBtn = document.querySelector(".random-name");
    randomNameBtn.style.marginLeft = "0rem";

    inputBox.value = canvas.getAttribute("data-name");
  }

  if (randomNameButton) {
    randomNameButton.addEventListener("click", () => {
      fetchRandomName();
    });
  }

  async function fetchRandomName() {
    const response = await fetch("/random-name");
    const randomName = await response.text();
    inputBox.value = randomName.replace(/"/g, "");
  }
});
