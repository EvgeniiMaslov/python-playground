
import * as posenet from '@tensorflow-models/posenet';
import Stats from 'stats.js';

import {drawBoundingBox, drawKeypoints, drawSkeleton, isMobile, toggleLoadingUI, tryResNetButtonName, tryResNetButtonText, updateTryResNetButtonDatGuiCss} from './demo_util';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultQuantBytes = 2;

const defaultMobileNetMultiplier = isMobile() ? 0.50 : 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 500;

const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const guiState = {
  algorithm: 'multi-pose',
  input: {
    architecture: 'MobileNetV1',
    outputStride: defaultMobileNetStride,
    inputResolution: defaultMobileNetInputResolution,
    multiplier: defaultMobileNetMultiplier,
    quantBytes: defaultQuantBytes
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};


/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
let reps = 0;
let up = true;
let down = false;
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  // since images are being fed from a webcam, we want to feed in the
  // original image and then just flip the keypoints' x coordinates. If instead
  // we flip the image, then correcting left-right keypoint pairs requires a
  // permutation on all the keypoints.
  const flipPoseHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    // Begin monitoring code for frames per second
    stats.begin();

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    
    const pose = await guiState.net.estimatePoses(video, {
      flipHorizontal: flipPoseHorizontal,
      decodingMethod: 'single-person'
    });
    poses = poses.concat(pose);
    minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
    minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
    // console.log(pose);

    let leftHip = pose[0]['keypoints'][11];
    let leftAnkle = pose[0]['keypoints'][15];
    let nose = pose[0]['keypoints'][0];

    let hipAnkleDist;
    let hipNoseDist;

    const conf = guiState.singlePoseDetection.minPartConfidence;

    if (leftHip['score'] > conf && leftAnkle['score'] > conf && nose['score'] > conf) {
      hipAnkleDist = Math.sqrt(Math.pow(leftHip['position']['x'] - leftAnkle['position']['x'], 2) +
        + Math.pow(leftHip['position']['y'] - leftAnkle['position']['y'], 2));

      hipNoseDist = Math.sqrt(Math.pow(leftHip['position']['x'] - nose['position']['x'], 2) +
        + Math.pow(leftHip['position']['y'] - nose['position']['y'], 2));

      if (hipNoseDist/hipAnkleDist > 1.4 && up) {
        reps += 1;
        up = false;
        down = true;
      }
      else if (hipNoseDist/hipAnkleDist <= 1.4 && down) {
        down = false;
        up = true;
      }
      console.log(hipAnkleDist, hipNoseDist);
    }
    
    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });
    // document.getElementById('reps').innerHTML = reps;
    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  toggleLoadingUI(true);
  const net = await posenet.load({
    architecture: guiState.input.architecture,
    outputStride: guiState.input.outputStride,  
    inputResolution: guiState.input.inputResolution,
    multiplier: guiState.input.multiplier,
    quantBytes: guiState.input.quantBytes
  });
  toggleLoadingUI(false);

  guiState.net = net;

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  let timer = new Vue({
    el: '#app',
    data: {
      readyTime: 30,
      restTime: 10,
      workTime: 20,
      currentTime: 30,
      timer: null,
      repTimer: null,
      rep: reps,
      work: false,
      activeColor: 'green',
      fontSize: 50,
      status: 'Ready',
      sets: '',
    },
    mounted() {
      // this.startTimer(),
      this.refreshReps()
    },
    destroyed() {
      this.stopTimer()
    },
    methods: {
      start() {
        if (this.status == 'Ready') {
          this.startTimer()
        }
        else if (this.status == 'End') {
          this.status = 'Ready',
          this.currentTime = this.readyTime,
          this.startTimer()
        }
      },
      startTimer() {
        this.timer = setInterval(() => {
          this.currentTime--
        }, 1000)
        console.log(this.sets)
      },
      stopTimer() {
        clearTimeout(this.timer)
      },
      refreshReps() {
        this.repTimer = setInterval(() => {
          this.rep = reps
        }, 10)
      },
    },
    watch: {
      currentTime(time) {
        if (time === 0 && (this.status == 'Rest' || this.status == 'Ready')) {
          this.stopTimer(),
          this.currentTime = this.workTime,
          this.startTimer(),
          this.activeColor = 'red',
          this.status = 'Work'
        }
        else if (time === 0 && this.status == 'Work') {
          this.stopTimer(),
          this.currentTime = this.restTime,
          this.startTimer(),
          this.activeColor = 'green',
          this.status = 'Rest',
          this.sets--
        }
        if (this.sets == 0) {
          this.status = 'End',
          this.currentTime = 0,
          this.stopTimer()
        }
        
      },   
    },
  })

  detectPoseInRealTime(video, net);

}


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();



