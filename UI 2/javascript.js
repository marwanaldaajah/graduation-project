function checkboxClick(checkbox) {
	if (checkbox.checked) {
		checkbox.nextSibling.style.color = 'gray';
	} else {
		checkbox.nextSibling.style.color = 'black';
	}
}
