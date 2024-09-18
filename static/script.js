document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultElement = document.getElementById('prediction-result');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(form);

        // Convert FormData to a plain object
        const formObject = {};
        formData.forEach((value, key) => {
            formObject[key] = value;
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formObject)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Response data:', data);  // Log the response data

            if (data.prediction !== undefined) {
                resultElement.textContent = `Prediction: ${data.prediction}`;
            } else {
                resultElement.textContent = `Error: ${data.error || 'No prediction available'}`;
            }
        } catch (error) {
            console.error("Error:", error);
            resultElement.textContent = `Error: ${error.message}`;
        }
    });
});