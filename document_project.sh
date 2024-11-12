#!/bin/bash

# Output file
OUTPUT_FILE="project_documentation.md"

# Create or clear the output file
echo "# Project Documentation" > "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add project structure using tree
echo "## Project Structure" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
tree -I "node_modules|.git|.cache|__pycache__|*.pyc|generated_images|*.log" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add file contents
echo "## File Contents" >> "$OUTPUT_FILE"

# Find all relevant files, excluding certain directories and file types
find . -type f \
    ! -path "*/\.*" \
    ! -path "*/node_modules/*" \
    ! -path "*/__pycache__/*" \
    ! -path "*/generated_images/*" \
    ! -name "*.pyc" \
    ! -name "*.log" \
    ! -name "*.md" \
    ! -name "*.txt" \
    -print0 | while IFS= read -r -d '' file; do
    
    # Get file extension
    extension="${file##*.}"
    
    # Add file header
    echo -e "\n### $file" >> "$OUTPUT_FILE"
    echo '```'"$extension" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
done

echo "Documentation generated in $OUTPUT_FILE"