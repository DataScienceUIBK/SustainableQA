import re
import bisect

def split_content_by_words(content_lines: list[str], max_words: int) -> list[list[str]]:
    parts = []
    current_part = []
    current_word_count = 0

    for line in content_lines:
        words = line.split()
        num_words = len(words)
        
        # Keep empty lines as part of content, they contribute to the structure.
        # Their word count is 0, so they don't impact max_words check.
        if num_words == 0:
            current_part.append(line)
            continue 

        if current_word_count + num_words > max_words:
            if current_word_count == 0:
                # Add oversized line as its own part, even if it exceeds max_words
                parts.append([line])
            else:
                # Finalize current part and start a new one with the current line
                parts.append(current_part)
                current_part = [line]
                current_word_count = num_words
        else:
            current_part.append(line)
            current_word_count += num_words

    if current_part:
        parts.append(current_part)

    return parts

def chunk_markdown_with_line_numbers(markdown_text: str, max_words: int ) -> str:
    lines = markdown_text.splitlines()

    # Define the page marker pattern
    page_marker_pattern = re.compile(r'^{\s*(\d+)\s*}------------------------------------------------')

    # Extract page markers
    page_markers = []
    for idx, line in enumerate(lines):
        match = page_marker_pattern.match(line.strip())
        if match:
            page_num = int(match.group(1))
            page_markers.append((idx + 1, page_num))  # Line numbers start from 1

    # Helper function to get page number for a line
    def get_page_for_line(line_number):
        if not page_markers:
            return 1  # Default to page 1 if no page markers are found in the entire document.

        marker_lines, page_numbers = zip(*page_markers)
        idx = bisect.bisect_left(marker_lines, line_number)
        
        if idx == 0:
            # If line_number is before the first recorded page marker,
            # assume it's part of the first page (page 1) of the document.
            return 1
        
        # Otherwise, the line_number falls after or at marker_lines[idx-1].
        # So, it belongs to the page indicated by page_numbers[idx-1].
        return page_numbers[idx - 1]

    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    headers = []

    for idx, line in enumerate(lines):
        # Exclude page marker lines from being identified as headers to avoid conflicts
        if page_marker_pattern.match(line.strip()):
            continue
        match = header_pattern.match(line)
        if match:
            headers.append({
                'line_number': idx + 1,
                'header': line,
                'level': len(match.group(1))
            })

    # Add a dummy header at the end to ensure all content after the last real header is captured
    headers.append({
        'line_number': len(lines) + 1,
        'header': '', # Empty header as a boundary
        'level': 0    # Lowest level
    })

    chunks = []
    i = 0
    chunk_index = 1
    while i < len(headers) - 1:
        start_line = headers[i]['line_number']
        next_i = i + 1

        # This loop collects consecutive headers that might form a single logical header block
        # For example:
        # ## Main Title
        # ### Subtitle
        # Content...
        # Here, headers[i] is "Main Title", headers[next_i] is "Subtitle".
        while next_i < len(headers) - 1 and headers[next_i]['line_number'] == headers[next_i - 1]['line_number'] + 1:
            next_i += 1

        end_line = headers[next_i]['line_number'] - 1
        
        # Extract all lines for the current logical header block (including headers and content)
        # and remove any page markers from them.
        chunk_lines_with_markers = lines[start_line - 1 : end_line]
        chunk_lines_clean = [line for line in chunk_lines_with_markers if not page_marker_pattern.match(line.strip())]

        # Determine how many lines at the beginning of chunk_lines_clean are headers.
        # This uses the same logic as the outer header loop, ensuring consistency.
        header_lines_count = next_i - i 
        
        # Separate the actual content lines from the header lines
        content_lines = chunk_lines_clean[header_lines_count:]

        # Split the content into smaller parts if it exceeds the max_words limit
        content_word_count = sum(len(line.split()) for line in content_lines)
        if content_word_count > max_words:
            content_parts = split_content_by_words(content_lines, max_words)
        else:
            content_parts = [content_lines] if content_lines else []

        # Determine the page number for this header block based on its starting line in the original document
        page_num_for_header_block = get_page_for_line(start_line)

        if not content_parts:
            # Case for a header block with no substantial content (e.g., just headers or empty lines)
            current_chunk_prefix = f"\n\n--- Chunk {chunk_index} (page {page_num_for_header_block}, starts at line {start_line}) ---\n" if page_num_for_header_block is not None else f"\n\n--- Chunk {chunk_index} (starts at line {start_line}) ---\n"
            chunk_content = "\n".join(chunk_lines_clean).strip()
            chunks.append(current_chunk_prefix + chunk_content)
            chunk_index += 1
        else:
            # If the content was split into multiple parts, each part gets its own chunk
            for part_idx, part in enumerate(content_parts):
                # Re-generate the prefix for each sub-chunk using the current, incrementing chunk_index
                # and the page number/start line of the original header block.
                current_chunk_prefix = f"\n\n--- Chunk {chunk_index} (page {page_num_for_header_block}, starts at line {start_line}) ---\n" if page_num_for_header_block is not None else f"\n\n--- Chunk {chunk_index} (starts at line {start_line}) ---\n"
                
                # Each sub-chunk should start with the original headers of the block
                full_chunk_content_for_part = chunk_lines_clean[:header_lines_count] + part
                chunk_content = "\n".join(full_chunk_content_for_part).strip()
                chunks.append(current_chunk_prefix + chunk_content)
                chunk_index += 1 # Increment for the next sub-chunk

        i = next_i # Move to the next major header block

    return "\n".join(chunks)

# Example usage with word limit = 100
if __name__ == "__main__":
    with open("pdf_report_cleaned.md", "r", encoding="utf-8") as f:
        markdown_text = f.read()

    chunked_markdown = chunk_markdown_with_line_numbers(markdown_text, max_words=350)

    with open("chunked_output.md", "w", encoding="utf-8") as f:
        f.write(chunked_markdown)

    print("Chunking completed. Output saved to chunked_output.md")