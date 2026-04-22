// Hamburger nav toggle
const toggle = document.getElementById('navToggle');
const links  = document.querySelector('.nav-links');
if (toggle && links) {
  toggle.addEventListener('click', () => links.classList.toggle('open'));
}
