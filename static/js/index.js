window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/";
var NUM_INTERP_FRAMES = 100;

var interp_images_muscular = [];
var interp_images_long_torso = [];
var interp_images_petite = [];
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

    bulmaSlider.attach();

})
