document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const queryForm = document.getElementById('queryForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const queryResult = document.getElementById('queryResult');

    // Handle PDF Upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const fileInput = document.getElementById('pdfFile');
        formData.append('file', fileInput.files[0]);
        
        uploadStatus.className = 'status';
        uploadStatus.textContent = 'Uploading...';
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                uploadStatus.className = 'status success';
                uploadStatus.textContent = `Success! ${data.message}`;
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.className = 'status error';
            uploadStatus.textContent = `Error: ${error.message}`;
        }
    });

    // Handle Query Submission
    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const prompt = document.getElementById('queryInput').value;
        queryResult.textContent = 'Processing query...';
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `prompt=${encodeURIComponent(prompt)}`
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Handle different types of content in the response
                if (data.response) {
                    queryResult.innerHTML = formatResponse(data.response);
                } else {
                    throw new Error('No response received');
                }
            } else {
                throw new Error(data.error || 'Query failed');
            }
        } catch (error) {
            queryResult.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        }
    });

    // Helper function to format response with tables and images
    function formatResponse(response) {
        let formattedResponse = response;
        
        // Handle tables (if response contains table markup)
        if (response.includes('<table>')) {
            // The response already contains HTML table markup
            formattedResponse = response;
        }
        
        // Handle images (if response contains base64 images)
        if (response.includes('data:image')) {
            const imgRegex = /data:image\/[^;]+;base64,[^"]+/g;
            formattedResponse = response.replace(imgRegex, match => 
                `<img src="${match}" class="result-image" alt="Retrieved image">`
            );
        }
        
        return formattedResponse;
    }
}); 