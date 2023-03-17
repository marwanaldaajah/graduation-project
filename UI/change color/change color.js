
document.querySelectorAll('input[type="checkbox"]').forEach(function(checkbox) {
  checkbox.addEventListener('change', function() {
    if (this.checked) {
      this.nextElementSibling.style.color = 'green';
      } 
      else
       {
      this.nextElementSibling.style.color = '';
      }
  });
});
