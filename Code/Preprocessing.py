import re

def is_heading(line: str) -> bool:
    """Checks if a line is a markdown heading (#, ##, ###, etc.)."""
    return bool(re.match(r'^\s*#{1,6}\s', line))

def is_page_marker(line: str) -> bool:
    """Checks if a line is a page marker in the form '{n}------' or '{n}======'."""
    return bool(re.match(r'^\{\d+\}[-=]+', line))

def is_table_line(line: str) -> bool:
    """Checks if a line is part of a markdown table."""
    if re.match(r'^\s*\|.*\|\s*$', line):
        return True
    if re.match(r'^\s*(:?-+:?\s*\|)+', line):
        return True
    return False

def is_image_line(line: str) -> bool:
    """Checks if a line contains a Markdown image or an HTML <img> tag."""
    if re.search(r'!\[.*?\]\(.*?\)', line):
        return True
    if re.search(r'<img\s+.*?>', line):
        return True
    return False

def is_footnote(line: str) -> bool:
    """Checks if a line looks like a standalone footnote."""
    return bool(re.match(r'^\s*(\^|(\[\d+\])|(\d+))\s', line))

def is_short_standalone_paragraph(line: str, word_threshold: int = 20, next_lines: list[str] = None) -> bool:
    """Checks if a line is a short standalone paragraph (< word_threshold words).
    Preserves lines ending with a colon followed by a list."""
    stripped = line.strip()
    words = stripped.split()
    if 0 < len(words) < word_threshold:
        if next_lines and stripped.endswith(':'):
            # Check if any subsequent non-blank line is a list item
            for next_line in next_lines:
                if next_line.strip() == '':
                    continue
                if is_list_item(next_line):
                    return False
                break
        return True
    return False

def is_list_item(line: str) -> bool:
    """Checks if a line is a markdown list item (e.g., '- text', '1. text')."""
    return bool(re.match(r'^\s*[-+*]\s', line)) or bool(re.match(r'^\s*\d+[\.\)]\s', line))

def process_line(line: str, next_lines: list[str]) -> str or None:
    """Removes unwanted lines like footnotes, tables, images, and short paragraphs."""
    if is_footnote(line) or is_table_line(line) or is_image_line(line):
        return None
    if is_page_marker(line) or is_heading(line) or is_list_item(line):
        return line
    if is_short_standalone_paragraph(line, next_lines=next_lines):
        return None
    return line

def is_real_content(line: str) -> bool:
    """Determines if a line is 'real content' (not blank, not a marker or heading)."""
    text = line.strip()
    return bool(text) and not is_page_marker(line) and not is_heading(line)

def remove_empty_sections(page_lines: list[str]) -> list[str]:
    """Removes headings with no real content until the next page boundary."""
    output = []
    i = 0
    n = len(page_lines)
    while i < n:
        line = page_lines[i]
        if is_heading(line):
            output.append(line)
            j = i + 1
            block_content = []
            while j < n and not (is_heading(page_lines[j]) or is_page_marker(page_lines[j])):
                block_content.append(page_lines[j])
                j += 1
            if (j >= n or is_page_marker(page_lines[j])) and not any(is_real_content(ln) for ln in block_content):
                output.pop()
            else:
                output.extend(block_content)
            i = j
        else:
            output.append(line)
            i += 1
    return output

def page_is_empty_after_cleaning(page_lines: list[str]) -> bool:
    """Checks if a page has no real content after cleaning."""
    return not any(is_real_content(ln) for ln in page_lines)

def collapse_blank_lines(lines: list[str]) -> list[str]:
    """Ensures no more than one consecutive blank line."""
    collapsed = []
    prev_blank = False
    for ln in lines:
        if ln.strip() == '':
            if not prev_blank:
                collapsed.append('')
                prev_blank = True
        else:
            collapsed.append(ln)
            prev_blank = False
    return collapsed

def remove_blank_lines_between_headings(lines: list[str]) -> list[str]:
    """Removes blank lines directly between two headings."""
    output = []
    n = len(lines)
    for i in range(n):
        if lines[i].strip() == '' and i > 0 and i < n - 1 and is_heading(lines[i - 1]) and is_heading(lines[i + 1]):
            continue
        output.append(lines[i])
    return output

def get_last_non_blank_line(page_lines: list[str]) -> str or None:
    """Returns the last non-blank line in a page, or None if all lines are blank."""
    for line in reversed(page_lines):
        if line.strip() != '':
            return line
    return None

def get_first_content_line_after_marker(page_lines: list[str]) -> str or None:
    """Returns the first non-blank line after the page marker, or None if none exists."""
    if not page_lines:
        return None
    for line in page_lines[1:]:
        if line.strip() != '':
            return line
    return None

def remove_empty_lines_before_list(page_lines: list[str]) -> list[str]:
    """Removes empty lines between a statement ending with a colon and a list."""
    output = []
    i = 0
    n = len(page_lines)
    while i < n:
        line = page_lines[i]
        output.append(line)
        # Check if the current line ends with a colon
        if line.strip().endswith(':'):
            j = i + 1
            empty_lines = []
            # Collect empty lines
            while j < n and page_lines[j].strip() == '':
                empty_lines.append(j)
                j += 1
            # Check if the next non-blank line is a list item
            if j < n and is_list_item(page_lines[j]):
                # Skip adding the empty lines to output
                i = j
                output.append(page_lines[j])
            else:
                # Add any empty lines if not followed by a list
                output.extend(page_lines[i+1:j])
                i = j
        else:
            i += 1
    return output

def process_markdown(input_path: str, output_path: str):
    """
    Processes a Markdown file by:
    1. Splitting into pages at page markers.
    2. Removing empty lines between a statement ending with a colon and a list.
    3. Cleaning lines (removing footnotes, tables, images, short paragraphs).
    4. Removing empty heading sections.
    5. Discarding pages with no real content.
    6. Collapsing blank lines.
    7. Removing blank lines between headings.
    8. Removing a heading at the bottom of a page if the next page starts with a heading.
    9. Writing the result to an output file.
    """
    pages = []
    current_page = []

    def finalize_page():
        if current_page:
            pages.append(current_page[:])
            current_page.clear()

    # Step 1: Split by page markers
    with open(input_path, 'r', encoding='utf-8') as infile:
        for raw_line in infile:
            line = raw_line.rstrip('\n')
            if is_page_marker(line):
                finalize_page()
                current_page.append(line)
            else:
                current_page.append(line)
        finalize_page()

    processed_pages = []

    # Process each page
    for page_lines in pages:
        # Step 2: Remove empty lines between statement and list
        cleaned = remove_empty_lines_before_list(page_lines)
        # Step 3: Clean lines, passing remaining lines for context
        cleaned = [process_line(cleaned[i], cleaned[i+1:] if i < len(cleaned)-1 else []) 
                   for i in range(len(cleaned)) if process_line(cleaned[i], cleaned[i+1:] if i < len(cleaned)-1 else []) is not None]
        # Step 4: Remove empty sections
        cleaned = remove_empty_sections(cleaned)
        # Step 5: Discard empty pages
        if page_is_empty_after_cleaning(cleaned):
            continue
        # Step 6: Collapse blank lines
        collapsed = collapse_blank_lines(cleaned)
        # Step 7: Remove blank lines between headings
        final_lines = remove_blank_lines_between_headings(collapsed)
        processed_pages.append(final_lines)

    # Step 8: Remove headings at page boundaries
    for i in range(len(processed_pages) - 1):
        last_line = get_last_non_blank_line(processed_pages[i])
        first_line_next = get_first_content_line_after_marker(processed_pages[i+1])
        if last_line is not None and first_line_next is not None and is_heading(last_line) and is_heading(first_line_next):
            for j in range(len(processed_pages[i]) - 1, -1, -1):
                if processed_pages[i][j].strip() != '':
                    del processed_pages[i][j]
                    break

    # Step 9: Write the output
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for page_content in processed_pages:
            for line in page_content:
                outfile.write(line + '\n')

    print(f"Done! Processed markdown successfully written to: {output_path}")

# Example usage:
if __name__ == "__main__":
    process_markdown("pdf_report.md", "pdf_report_cleaned.md")