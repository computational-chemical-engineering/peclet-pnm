#include "sdf_reader.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// A simple string helper to find value of an XML attribute
// e.g. WholeExtent="0 10 0 10 0 10"
std::string get_attribute_value(const std::string &content,
                                const std::string &attr_name) {
  size_t pos = content.find(attr_name + "=\"");
  if (pos == std::string::npos)
    return "";

  size_t start = pos + attr_name.length() + 2;
  size_t end = content.find("\"", start);
  if (end == std::string::npos)
    return "";

  return content.substr(start, end - start);
}

SDFData SDFReader::read_vti(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  SDFData data;

  // Read file content line by line until we find AppendedData or similar
  // We assume the header is small enough to read comfortably
  // BUT we need to be careful not to read the binary part as text
  // Strategy: Read chunks or just locate the "<AppendedData encoding=\"raw\">"
  // tag.

  // A clearer approach for mixed text/binary:
  // Read the file into a string buffer until we see the start of binary marker.
  // The user said: "Appended Raw Binary".
  // Usually VTK VTI structure is:
  // <VTKFile ...> ... <AppendedData
  // encoding="raw">_LENGTH_DATA...</AppendedData></VTKFile>

  // We will scan for the underscore `_` identifying the start of raw data,
  // but we must first parse the header info which appears before.

  std::string header_buffer;
  char c;
  bool found_marker = false;

  // We'll read until we see the sequence `_` immediately following
  // "<AppendedData encoding=\"raw\">" Actually, often it's just
  // `...<AppendedData encoding="raw">` then newline then `_`. Let's read the
  // first 4KB, it should contain the header.

  // Better: Read whole file into memory? No, could be large.
  // Let's read until "</InAppendedData>" is not useful because it's at the end.
  // We search for the offset=0 in the AppendedData.

  // Implementation:
  // 1. Read header text until `header_type="UInt64"` (verification) or just
  // parse specific tags.
  // 2. Find "WholeExtent", "Origin", "Spacing".

  // Let's read a chunk (e.g. 8KB) to parse XML.
  std::vector<char> buffer(8192);
  file.read(buffer.data(), buffer.size());
  std::string header(buffer.data(), file.gcount());

  // Parse WholeExtent
  std::string extent_str = get_attribute_value(header, "WholeExtent");
  if (extent_str.empty())
    throw std::runtime_error("Could not find WholeExtent in VTI header");

  int x1, x2, y1, y2, z1, z2;
  std::stringstream ss_ext(extent_str);
  ss_ext >> x1 >> x2 >> y1 >> y2 >> z1 >> z2;
  data.resolution = {x2 - x1 + 1, y2 - y1 + 1, z2 - z1 + 1};

  // Parse Origin
  std::string origin_str = get_attribute_value(header, "Origin");
  std::stringstream ss_org(origin_str);
  ss_org >> data.origin[0] >> data.origin[1] >> data.origin[2];

  // Parse Spacing
  std::string spacing_str = get_attribute_value(header, "Spacing");
  std::stringstream ss_spc(spacing_str);
  ss_spc >> data.spacing[0] >> data.spacing[1] >> data.spacing[2];

  // Find binary data start
  // Look for "<AppendedData encoding=\"raw\">"
  std::string appended_tag = "<AppendedData encoding=\"raw\">";
  size_t tag_pos = header.find(appended_tag);
  if (tag_pos == std::string::npos)
    throw std::runtime_error("Could not find AppendedData tag");

  // The binary data starts after the tag. It typically starts with an
  // underscore `_` followed by the length (UInt64). Let's search for `_` after
  // the tag.
  size_t underscore_pos = header.find("_", tag_pos);
  if (underscore_pos == std::string::npos) {
    // Maybe we didn't read enough?
    // If the header is larger than 8KB, we are in trouble.
    // Realistically headers are small.
    throw std::runtime_error(
        "Could not find binary data marker '_' in first 8KB");
  }

  // Seek file to the position of the length (immediately after `_`)
  file.clear(); // Clear EOF flag if we read past end in buffer
  file.seekg(underscore_pos + 1, std::ios::beg);

  // Read length (UInt64)
  uint64_t data_bytes = 0;
  file.read(reinterpret_cast<char *>(&data_bytes), sizeof(uint64_t));

  // Check for read failure
  if (!file) {
    throw std::runtime_error("Failed to read data_bytes (UInt64) from file.");
  }

  std::cout << "Read data length: " << data_bytes << std::endl;

  size_t attempts = 0;
  // Check if expected size matches our resolution
  // Float32 = 4 bytes
  size_t expected_bytes = data.size() * sizeof(float);
  if (data_bytes != expected_bytes) {
    std::cerr << "Warning: Data length in file (" << data_bytes
              << ") does not match expected size from resolution ("
              << expected_bytes << ")." << std::endl;
    // Proceeding anyway but this is suspicious.
    // It might be that the file size includes some padding or we misread
    // resolution. Or maybe its compressed (zlib)? User said: "Appended Raw
    // Binary". This implies uncompressed.
  }

  // Read Data
  data.sdf_values.resize(data.size());
  file.read(reinterpret_cast<char *>(data.sdf_values.data()), data_bytes);

  if (file.gcount() != static_cast<std::streamsize>(data_bytes)) {
    std::cerr << "Warning: Could not read all data. Read " << file.gcount()
              << " bytes." << std::endl;
  }

  return data;
}
