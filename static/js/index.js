window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/";
var NUM_INTERP_FRAMES = 100;

var interp_images_muscular = [];
var interp_images_long_torso = [];
var interp_images_petite = [];
var interp_images_smiling = [];
var interp_images_open_mouth = [];
var interp_images_serious = [];
var interp_images_ears_out = [];
var interp_images_fat = [];
var interp_images_long_neck = [];
var interp_images_lion = [];
var interp_images_cow = [];
var interp_images_cat = [];
var interp_images_small = [];
var interp_images_thin = [];
var interp_images_masculin = [];

function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "long_torso_interpolation" + '/' + String(i) + '.png';
    interp_images_long_torso[i] = new Image();
    interp_images_long_torso[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "muscular_interpolation" + '/' + String(i) + '.png';
    interp_images_muscular[i] = new Image();
    interp_images_muscular[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "petite_interpolation" + '/' + String(i) + '.png';
    interp_images_petite[i] = new Image();
    interp_images_petite[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "serious_interpolation" + '/' + String(i) + '.png';
    interp_images_serious[i] = new Image();
    interp_images_serious[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "smiling_interpolation" + '/' + String(i) + '.png';
    interp_images_smiling[i] = new Image();
    interp_images_smiling[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "open_mouth_interpolation" + '/' + String(i) + '.png';
    interp_images_open_mouth[i] = new Image();
    interp_images_open_mouth[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "fat_interpolation" + '/' + String(i) + '.png';
    interp_images_fat[i] = new Image();
    interp_images_fat[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "ears_sticking_out_interpolation" + '/' + String(i) + '.png';
    interp_images_ears_out[i] = new Image();
    interp_images_ears_out[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "long_neck_interpolation" + '/' + String(i) + '.png';
    interp_images_long_neck[i] = new Image();
    interp_images_long_neck[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "lion_interpolation" + '/' + String(i) + '.png';
    interp_images_lion[i] = new Image();
    interp_images_lion[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "cow_interpolation" + '/' + String(i) + '.png';
    interp_images_cow[i] = new Image();
    interp_images_cow[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "cat_interpolation" + '/' + String(i) + '.png';
    interp_images_cat[i] = new Image();
    interp_images_cat[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "thin_interpolation" + '/' + String(i) + '.png';
    interp_images_thin[i] = new Image();
    interp_images_thin[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "masculin_interpolation" + '/' + String(i) + '.png';
    interp_images_masculin[i] = new Image();
    interp_images_masculin[i].src = path;
  }
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + "small_interpolation" + '/' + String(i) + '.png';
    interp_images_small[i] = new Image();
    interp_images_small[i].src = path;
  }
}

function setInterpolationImage_muscular(i) {
  var image = interp_images_muscular[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-muscular').empty().append(image);
}
function setInterpolationImage_long_torso(i) {
  var image = interp_images_long_torso[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-long-torso').empty().append(image);
}
function setInterpolationImage_petite(i) {
  var image = interp_images_petite[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-petite').empty().append(image);
}
function setInterpolationImage_smiling(i) {
  var image = interp_images_smiling[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-smiling').empty().append(image);
}
function setInterpolationImage_serious(i) {
  var image = interp_images_serious[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-serious').empty().append(image);
}
function setInterpolationImage_open_mouth(i) {
  var image = interp_images_open_mouth[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-open-mouth').empty().append(image);
}
function setInterpolationImage_fat(i) {
  var image = interp_images_fat[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-fat').empty().append(image);
}
function setInterpolationImage_ears_out(i) {
  var image = interp_images_ears_out[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-ears-out').empty().append(image);
}
function setInterpolationImage_long_neck(i) {
  var image = interp_images_long_neck[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-long-neck').empty().append(image);
}
function setInterpolationImage_lion(i) {
  var image = interp_images_lion[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-lion').empty().append(image);
}
function setInterpolationImage_cow(i) {
  var image = interp_images_cow[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-cow').empty().append(image);
}
function setInterpolationImage_cat(i) {
  var image = interp_images_cat[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-cat').empty().append(image);
}
function setInterpolationImage_small(i) {
  var image = interp_images_small[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-small').empty().append(image);
}
function setInterpolationImage_masculin(i) {
  var image = interp_images_masculin[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-masculin').empty().append(image);
}
function setInterpolationImage_thin(i) {
  var image = interp_images_thin[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-thin').empty().append(image);
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider-muscular').on('input', function(event) {
      setInterpolationImage_muscular(this.value);
    });
    setInterpolationImage_muscular(50);
    $('#interpolation-slider-muscular').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-long-torso').on('input', function(event) {
      setInterpolationImage_long_torso(this.value);
    });
    setInterpolationImage_long_torso(50);
    $('#interpolation-slider-long-torso').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-petite').on('input', function(event) {
      setInterpolationImage_petite(this.value);
    });
    setInterpolationImage_petite(50);
    $('#interpolation-slider-petite').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-smiling').on('input', function(event) {
      setInterpolationImage_smiling(this.value);
    });
    setInterpolationImage_smiling(50);
    $('#interpolation-slider-smiling').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-serious').on('input', function(event) {
    setInterpolationImage_serious(this.value);
    });
    setInterpolationImage_serious(50);
    $('#interpolation-slider-serious').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-open-mouth').on('input', function(event) {
      setInterpolationImage_open_mouth(this.value);
    });
    setInterpolationImage_open_mouth(50);
    $('#interpolation-slider-open-mouth').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-fat').on('input', function(event) {
      setInterpolationImage_fat(this.value);
    });
    setInterpolationImage_fat(50);
    $('#interpolation-slider-fat').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-ears-out').on('input', function(event) {
      setInterpolationImage_ears_out(this.value);
    });
    setInterpolationImage_ears_out(50);
    $('#interpolation-slider-ears-out').prop('max', NUM_INTERP_FRAMES - 1);

    $('#interpolation-slider-long-neck').on('input', function(event) {
      setInterpolationImage_long_neck(this.value);
    });
    setInterpolationImage_long_neck(50);
    $('#interpolation-slider-long-neck').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

    $('#interpolation-slider-cat').on('input', function(event) {
      setInterpolationImage_cat(this.value);
    });
    setInterpolationImage_cat(50);
    $('#interpolation-slider-cat').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

    $('#interpolation-slider-lion').on('input', function(event) {
      setInterpolationImage_lion(this.value);
    });
    setInterpolationImage_lion(50);
    $('#interpolation-slider-lion').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

    $('#interpolation-slider-cow').on('input', function(event) {
      setInterpolationImage_cow(this.value);
    });
    setInterpolationImage_cow(50);
    $('#interpolation-slider-cow').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

    $('#interpolation-slider-thin').on('input', function(event) {
      setInterpolationImage_thin(this.value);
    });
    setInterpolationImage_thin(50);
    $('#interpolation-slider-thin').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

    $('#interpolation-slider-small').on('input', function(event) {
      setInterpolationImage_small(this.value);
    });
    setInterpolationImage_small(50);
    $('#interpolation-slider-small').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

    $('#interpolation-slider-masculin').on('input', function(event) {
      setInterpolationImage_masculin(this.value);
    });
    setInterpolationImage_masculin(50);
    $('#interpolation-slider-masculin').prop('max', NUM_INTERP_FRAMES - 1);
    bulmaSlider.attach();

})
