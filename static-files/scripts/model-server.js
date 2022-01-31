class ModelServerFrontEnd {
   constructor({ form, submitButton, fileInput, hiddenError, previewImg,
      fileUploadText, imagePane, svgDiv, successEl, predictionsEl,
      filename, filesize, filetype }) {
      this._svgDiv = svgDiv;
      this._currentSvg = null;
      this._scaleDiff = 0;
      this._viewBox = null;
      this._data = null;

      this._form = form;
      this._submitButton = submitButton;
      this._fileInput = fileInput;
      this._hiddenError = hiddenError;
      this._previewImg = previewImg;
      this._fileUploadText = fileUploadText;
      this._imagePane = imagePane;
      this._successEl = successEl;
      this._predictionsEl = predictionsEl;
      this._filename = filename;
      this._filetype = filetype;
      this._filesize = filesize;

   }

   validateAndSubmit = (e) => {
      e.preventDefault();
      const file = this._fileInput.files[0];
      const validation = this.validateFile(file);
      this._hiddenError.innerHTML = `${validation.message}`;
      if (validation.ok) {
         return this.postFile();
      }
   }



   preview = (file) => {
      // new file reset results
      this._predictionsEl.innerHTML = "";
      this._successEl.innerHTML = "";
      this._data = null;
      this._imagePane.hidden = true;
      this._previewImg.src = "";
      this._filename.innerHTML = "";
      this._filesize.innerHTML = "";
      this._filetype.innerHTML = "";
      this._svgDiv.innerHTML = "";
      this._imagePane.hidden = true;

      if (file) {
         this._filename.innerHTML = file.name;
         this._filesize.innerHTML = file.size;
         this._filetype.innerHTML = file.type;
         this._previewImg.src = URL.createObjectURL(file);

         this._previewImg.onload = (e) => {
            this._imagePane.hidden = false;
            console.log("New preview loaded.....")
            // this._scaleDiff = this._previewImg.width / this._previewImg.naturalWidth;
            this._viewBox = { width: this._previewImg.naturalWidth, height: this._previewImg.naturalHeight };
            // this._svgDiv.setAttribute("style", `width: ${this._viewBox.width}px; height: ${this._viewBox.height}px`);
            // this._imagePane.setAttribute("style", `width: ${this._viewBox.width}px; height: ${this._viewBox.height}px`);
         }

      }

   }

   validateFile = (file) => {
      if (file) {
         this._submitButton.disabled = false;
         return { ok: true, message: "" };
      } else {
         this._submitButton.disabled = true;
         return { ok: false, message: "Error with file" };
      }
   }

   postFile = () => {
      // FOR TESTING:: define & return this.handleData(fakeData);
      var myHeaders = new Headers();
      myHeaders.append("Accept", "*/*");
      myHeaders.append("Connection", "keep-alive");
      myHeaders.append("Host", "https://adamant.tator.io/");
      /* NOTE: Do NOT send content-type */ // myHeaders.append("Content-Type", 'multipart/form-data;boundary=""');

      var formdata = new FormData();
      formdata.append("file", this._fileInput.files[0]);

      var requestOptions = {
         method: 'POST',
         headers: myHeaders,
         body: formdata,
         redirect: 'follow'
      };

      this._successEl.innerHTML = "Loading....";
      fetch("https://adamant.tator.io:8082/predictor/", requestOptions)
         .then(response => response.json())
         .then(data => {
            console.log(data);
            return this.handleData(data);
         })
         .catch(error => console.error('Could not fetch predictions.', error));
   }


   handleResize = () => {
      if (this._data !== null) {
         this.handleData(this._data);
      }
   }

   handleData = (data) => {
      // console.log(data);
      this._data = data;
      if (data.success !== null && data.success == true && data.predictions !== null) {
         this._successEl.innerHTML = data.predictions.length;
         // this._filename.innerHTML = "";
         // this._filesize.innerHTML = "";
         // this._filetype.innerHTML = "";
      } else {
         this._successEl.innerHTML = "Error";
         this._data = null;
      }
      if (data.predictions !== null) {
         this._predictionsEl.innerHTML = "";
         // conceptsSVG.innerHTML = "";

         const viewBoxSize = `0 0 ${this._viewBox.width} ${this._viewBox.height}`;
         // reset and size concepts svg canvas
         // conceptsSVG.innerHTML = "";

         this._currentSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");


         this._currentSvg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
         // this._currentSvg.setAttribute("class", "concepts-figure__svg image-details__image");
         this._currentSvg.setAttribute("viewBox", viewBoxSize);
         // this._currentSvg.setAttribute("style", `height: ${this._viewBox.height}px; width: ${this._viewBox.width}px;`)
         // this._currentSvg.setAttribute("height", this._viewBox.height);
         // this._currentSvg.setAttribute("width", this._viewBox.width);

         for (let prediction of data.predictions) {
            let predictionNode = document.createElement("li");
            this._predictionsEl.appendChild(predictionNode);

            const details = document.createElement("details");
            details.setAttribute("style", "padding-bottom: 15px;")
            predictionNode.appendChild(details);

            let categoryId = document.createElement("summary");
            categoryId.setAttribute("class", 'meta-box__link link');
            categoryId.setAttribute("style", "font-weight: 700");

            let scores = document.createElement("div");
            scores.setAttribute("style", "padding-left: 20px; color: rgb(109, 108, 144);");

            let bbox = document.createElement("div");
            bbox.setAttribute("style", "padding-left: 20px; color: rgb(109, 108, 144);");


            if (prediction.category_id !== null) {
               categoryId.textContent = `Category ID: ${prediction.category_id}`;
               details.appendChild(categoryId);
            }

            if (prediction.scores !== null) {
               scores.innerHTML = "<strong class='text-meta'>Score:</strong> ";
               for (let score of prediction.scores) {
                  scores.innerHTML += `<span style="">${score}<span>`;
               }
               details.appendChild(scores);
            }

            if (prediction.bbox !== null) {
               try {
                  details.appendChild(bbox);
                  this.showBoundingBoxes(bbox, prediction);
               } catch (err) {
                  console.log("Unable to draw bounding box.", err);
               }
            }


         }

         this._svgDiv.innerHTML = this._currentSvg.outerHTML;
      } else {
         this._predictionsEl.innerHTML = "N/A";
      }
   }

   showBoundingBoxes(bbox, prediction) {
      // console.log(prediction);
      this._strokeWidth = Math.min(this._previewImg.naturalHeight, 1080) / 200;
      this._fontSize = Math.min(this._previewImg.naturalHeight, 1080) / 333 * 12;
      const strokeWidth = this._strokeWidth;
      const boundingBox = JSON.parse(prediction.bbox);
      const bounding_x1 = Number(boundingBox[0]) + (strokeWidth / 2);
      const bounding_y1 = Number(boundingBox[1]) + (2 * strokeWidth);
      const bounding_x2 = Number(boundingBox[2]) + (strokeWidth / 2);
      const bounding_y2 = Number(boundingBox[3]) + (2 * strokeWidth);
      const transformPoints =

         bbox.innerHTML = `<strong class='text-meta'>Bounding Boxes:</strong><br/><div class="label-1>`;
      bbox.innerHTML += ` &nbsp; <strong>x1</strong>: ${boundingBox[0]}<br/>`;
      bbox.innerHTML += ` &nbsp; <strong>y1</strong>: ${boundingBox[1]}<br/>`;
      bbox.innerHTML += ` &nbsp; <strong>x2</strong>: ${boundingBox[2]}<br/>`;
      bbox.innerHTML += ` &nbsp; <strong>y2</strong>: ${boundingBox[3]}</div>`;


      const bounding_width = bounding_x2 - bounding_x1;
      const bounding_height = bounding_y2 - bounding_y1;

      let hash = [...prediction.category_id].reduce((acc, char) => {
         return char.charCodeAt(0) + acc;
      }, 0);
      const colorHash = hash % 360;
      const colorString = `hsla(${colorHash}, 100%, 85%, 1)`;

      const box_G = document.createElement("g");
      box_G.setAttribute("id", `${String(prediction.category_id).replace(" ", "-")}__${bounding_x1}__${bounding_x1}`);
      box_G.setAttribute("style", `color: ${colorString}; stroke: ${colorString}; stroke-width: ${strokeWidth}px;`);
      box_G.setAttribute("x", `${bounding_x1}`);
      box_G.setAttribute("y", `${bounding_y1}`);
      box_G.setAttribute("transform", `translate(${bounding_x1}, ${bounding_y1})`);
      box_G.setAttribute("class", `concepts-figure__svg-group`);
      box_G.setAttribute("active", `true`);
      box_G.setAttribute("ref", `group`);
      this._currentSvg.appendChild(box_G);


      const concept_Text = document.createElement("text");
      // concept_Text.setAttribute("data-v-601a8666", "");
      concept_Text.setAttribute("style", `fill: ${colorString}; transform: translateY(1.25em); font-size: ${this._fontSize}px;`)
      concept_Text.setAttribute("stroke", `none`);
      concept_Text.setAttribute("fill", colorString);
      let labelX = this.labelXPosition(prediction.category_id.length, bounding_width, bounding_x1);
      concept_Text.setAttribute("x", labelX);
      let labelY = this.labelYPosition(bounding_height, bounding_y1);
      concept_Text.setAttribute("y", labelY);
      concept_Text.setAttribute("class", `concepts-figure__svg-text`);
      concept_Text.innerText = prediction.category_id;
      box_G.appendChild(concept_Text);


      const box_path = document.createElement("path");
      // box_path.setAttribute("data-v-601a8666", "");
      box_path.setAttribute("stroke", colorString);
      box_path.setAttribute("stroke-width", `${strokeWidth}px`);
      box_path.setAttribute("fill", "transparent");
      box_path.setAttribute("class", "concepts-figure__svg-shape");
      box_path.setAttribute("d", `m 0 0
                        h ${Math.floor(bounding_width)}
                        v ${Math.floor(bounding_height)}
                        h ${-Math.floor(bounding_width)}
                        z`);
      box_G.appendChild(box_path);
   }

   labelXPosition = (conceptLength, bounding_width, bounding_x) => {
      let labelHeight = this._fontSize; // font size calculation
      // let conceptLength = this.conceptName.length; //- (this.boundingBox.width/2)
      let conceptLengthSM = conceptLength * (labelHeight / 4);
      let conceptLengthLG = conceptLengthSM + (bounding_width);
      let space = (this._previewImg.width - bounding_x);

      if (conceptLengthLG >= space) {
         return Math.floor(-(conceptLengthSM)); //-conceptLength
      } else {
         return 0;
      }
   }

   labelYPosition = (bounding_height, bounding_y) => {
      let labelHeight = this._fontSize; // font size calculation
      let actualHeightNeeded = labelHeight + bounding_height;
      let needsMoreSpace = (bounding_y + actualHeightNeeded) > this._previewImg.height;

      if (needsMoreSpace) {
         return Math.floor(-(labelHeight + (this._strokeWidth * 4)));
      } else {
         return Math.floor(0 + (bounding_height - (this._strokeWidth * 4)));
      }
   }

   validateAndPreview = () => {
      const file = this._fileInput.files[0];
      this.validateFile(file);
      this.preview(file);
   }
}