var fakeData = {
   "success": true,
   "predictions": [
       {
           "category_id": "Glass sponge",
           "scores": [
               0.936853289604187
           ],
           "bbox": [
               221.26730346679688,
               95.90970611572266,
               275.6682434082031,
               158.79237365722656
           ]
       },
       {
           "category_id": "Glass sponge",
           "scores": [
               0.9360066652297974
           ],
           "bbox": [
               390.29974365234375,
               205.02911376953125,
               412.7304382324219,
               240.4307861328125
           ]
       },
       {
           "category_id": "Crab",
           "scores": [
               0.9176263809204102
           ],
           "bbox": [
               370.3782958984375,
               120.75619506835938,
               396.8592529296875,
               170.28118896484375
           ]
       },
       {
           "category_id": "Anemone",
           "scores": [
               0.8222666382789612
           ],
           "bbox": [
               338.8046875,
               240.5189666748047,
               358.1540222167969,
               262.97900390625
           ]
       },
       {
           "category_id": "Shrimp",
           "scores": [
               0.7843353748321533
           ],
           "bbox": [
               351.8642883300781,
               333.50933837890625,
               367.9346618652344,
               356.86712646484375
           ]
       },
       {
           "category_id": "Sea star",
           "scores": [
               0.6080804467201233
           ],
           "bbox": [
               273.8749694824219,
               191.44650268554688,
               295.8823547363281,
               219.82554626464844
           ]
       },
       {
           "category_id": "Anemone",
           "scores": [
               0.4814399182796478
           ],
           "bbox": [
               413.636474609375,
               97.76058197021484,
               425.3978271484375,
               108.77407836914062
           ]
       },
       {
           "category_id": "Anemone",
           "scores": [
               0.4343058168888092
           ],
           "bbox": [
               114.55255126953125,
               208.422119140625,
               129.5889892578125,
               221.33343505859375
           ]
       },
       {
           "category_id": "Shrimp",
           "scores": [
               0.31218352913856506
           ],
           "bbox": [
               202.02691650390625,
               99.27082824707031,
               221.85006713867188,
               110.91401672363281
           ]
       },
       {
           "category_id": "Crab",
           "scores": [
               0.272942453622818
           ],
           "bbox": [
               371.9002685546875,
               126.96249389648438,
               391.0743408203125,
               149.70510864257812
           ]
       },
       {
           "category_id": "Glass sponge",
           "scores": [
               0.21144063770771027
           ],
           "bbox": [
               229.55535888671875,
               107.83536529541016,
               288.4970703125,
               192.87518310546875
           ]
       },
       {
           "category_id": "Gastropod",
           "scores": [
               0.21023178100585938
           ],
           "bbox": [
               248.8347930908203,
               189.96746826171875,
               281.1795959472656,
               218.367431640625
           ]
       },
       {
           "category_id": "Anemone",
           "scores": [
               0.2003752887248993
           ],
           "bbox": [
               350.1575927734375,
               134.4268798828125,
               370.12908935546875,
               156.498779296875
           ]
       }
   ]
};
class ModelServerFrontEnd {
   constructor({ form, submitButton, fileInput, hiddenError, previewImg, imagePane, svgDiv, successEl, predictionsEl,
      filename, filesize, filetype, downloadResults, clearImage, uploadHeading }) {
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
      // this._fileUploadText = fileUploadText;
      this._imagePane = imagePane;
      this._successEl = successEl;
      this._predictionsEl = predictionsEl;
      this._filename = filename;
      this._filetype = filetype;
      this._filesize = filesize;
      this._downloadResults = downloadResults;
      this._clearImage = clearImage;
      this._uploadHeading = uploadHeading;

      this._downloadResults.addEventListener("click", this.downloadObjectAsJson.bind(this));
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
      this._downloadResults.hidden = true;
      this._clearImage.hidden = true;
      this._uploadHeading.hidden = false;

      if (file) {
         this._filename.innerHTML = file.name;
         this._filesize.innerHTML = file.size;
         this._filetype.innerHTML = file.type;
         this._previewImg.src = URL.createObjectURL(file);
         this._clearImage.hidden = false;
         this._uploadHeading.hidden = true;

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
      // FOR TESTING:: define &   
      console.log("Handling fake data.");
      console.log(fakeData);
      return this.handleData(fakeData);
      var myHeaders = new Headers();
      myHeaders.append("Accept", "*/*");
      myHeaders.append("Connection", "keep-alive");
      myHeaders.append("Host", "https://adamant.tator.io:8082/");
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
         this._downloadResults.hidden = false;
      } else {
         this._successEl.innerHTML = "Error";
         this._data = null;
         this._downloadResults.hidden = true;
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
         this._downloadResults.hidden = true;
      }
   }

   showBoundingBoxes(bbox, prediction) {
      // console.log(prediction);
      this._strokeWidth = Math.min(this._previewImg.naturalHeight, 1080) / 200;
      this._fontSize = Math.min(this._previewImg.naturalHeight, 1080) / 333 * 12;
      const strokeWidth = this._strokeWidth;
      const boundingBox = Array.isArray(prediction.bbox) ? prediction.bbox : JSON.parse(prediction.bbox);
      const bounding_x1 = Number(boundingBox[0]) + (strokeWidth / 2);
      const bounding_y1 = Number(boundingBox[1]) + (2 * strokeWidth);
      const bounding_x2 = Number(boundingBox[2]) + (strokeWidth / 2);
      const bounding_y2 = Number(boundingBox[3]) + (2 * strokeWidth);

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

   downloadObjectAsJson() {
      const exportObj = this._data;
      let date = new Date().toISOString();
      const exportName = `Results__${String(this._filename.innerText).split(".")[0]}__${date}`
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
      const downloadAnchorNode = document.createElement('a');

      downloadAnchorNode.setAttribute("href", dataStr);
      downloadAnchorNode.setAttribute("download", exportName + ".json");
      document.body.appendChild(downloadAnchorNode); // required for firefox
      
      downloadAnchorNode.click();
      downloadAnchorNode.remove();
    }
}