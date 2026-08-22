#ifndef NNPS_PARTICLE_BINARY_WRITER_CUH
#define NNPS_PARTICLE_BINARY_WRITER_CUH

#include "Points.cuh"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class ParticleBinaryWriter {
public:
    std::map<std::string, bool> print_info{
        {"id", true},
        {"m_h", true},
        {"level", true},
        {"neighbor_num", true}
    };

    explicit ParticleBinaryWriter(std::filesystem::path output_directory = "outdata")
        : output_directory_(std::move(output_directory)) {}

    void Output_particle(int n_out, const Points& points) {
        std::filesystem::create_directories(output_directory_);
        Output_vtu_file(n_out, points);
        Output_pvtu_file(n_out, points);
        Output_pvd_file(n_out);
    }

    void Output_vtu_file(int n_out, const Points& points) const {
        const std::filesystem::path filename =
            output_directory_ / ("particle_step_" + std::to_string(n_out) + ".vtu");
        std::ofstream vtu_file(filename, std::ios::binary | std::ios::trunc);
        if (!vtu_file) {
            return;
        }

        const std::size_t particle_count = points.particle_num;
        std::ostringstream appended_data(std::ios::binary);
        std::vector<std::string> point_data_arrays;
        std::uint32_t offset = 0;

        if (print_info.at("id")) {
            point_data_arrays.push_back(data_array_header(
                "Int32", "id", offset));
            append_ids(appended_data, particle_count, offset);
        }
        if (print_info.at("m_h")) {
            point_data_arrays.push_back(data_array_header(
                "Float32", "m_h", offset));
            append_values(appended_data, points.host_h, particle_count, offset);
        }
        if (print_info.at("level")) {
            point_data_arrays.push_back(data_array_header(
                "Int32", "level", offset));
            append_values(appended_data, points.host_level, particle_count, offset);
        }
        if (print_info.at("neighbor_num")) {
            point_data_arrays.push_back(data_array_header(
                "UInt32", "neighbor_num", offset));
            append_values(appended_data, points.host_neighbor_num, particle_count, offset);
        }

        const std::uint32_t points_offset = offset;
        append_coordinates(appended_data, points, offset);

        const std::uint32_t connectivity_offset = offset;
        append_connectivity(appended_data, particle_count, offset);
        const std::uint32_t offsets_offset = offset;
        append_offsets(appended_data, particle_count, offset);
        const std::uint32_t types_offset = offset;
        append_cell_types(appended_data, particle_count, offset);

        vtu_file << "<?xml version=\"1.0\"?>\n"
                    "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                    "byte_order=\"LittleEndian\">\n"
                    " <UnstructuredGrid>\n"
                 << "  <Piece NumberOfPoints=\"" << particle_count
                 << "\" NumberOfCells=\"" << particle_count << "\">\n"
                    "   <PointData>\n";

        for (const std::string& array_header : point_data_arrays) {
            vtu_file << array_header;
        }

        vtu_file << "   </PointData>\n"
                    "   <Points>\n"
                    "    <DataArray type=\"Float32\" Name=\"Points\" "
                    "NumberOfComponents=\"3\" format=\"appended\" offset=\""
                 << points_offset << "\"/>\n"
                    "   </Points>\n"
                    "   <Cells>\n"
                    "    <DataArray type=\"Int32\" Name=\"connectivity\" "
                    "format=\"appended\" offset=\""
                 << connectivity_offset << "\"/>\n"
                    "    <DataArray type=\"Int32\" Name=\"offsets\" "
                    "format=\"appended\" offset=\""
                 << offsets_offset << "\"/>\n"
                    "    <DataArray type=\"UInt8\" Name=\"types\" "
                    "format=\"appended\" offset=\""
                 << types_offset << "\"/>\n"
                    "   </Cells>\n"
                    "  </Piece>\n"
                    " </UnstructuredGrid>\n"
                    " <AppendedData encoding=\"raw\">\n_";

        const std::string raw_data = appended_data.str();
        vtu_file.write(raw_data.data(), static_cast<std::streamsize>(raw_data.size()));
        vtu_file << "\n </AppendedData>\n</VTKFile>\n";
    }

    void Output_pvtu_file(int n_out, const Points&) const {
        const std::filesystem::path filename =
            output_directory_ / ("particle_step_" + std::to_string(n_out) + ".pvtu");
        std::ofstream pvtu_file(filename, std::ios::trunc);
        if (!pvtu_file) {
            return;
        }

        pvtu_file << "<?xml version=\"1.0\"?>\n"
                     "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" "
                     "byte_order=\"LittleEndian\">\n"
                     " <PUnstructuredGrid GhostLevel=\"0\">\n"
                     "  <PPointData>\n";

        if (print_info.at("id")) {
            pvtu_file << "   <PDataArray type=\"Int32\" Name=\"id\"/>\n";
        }
        if (print_info.at("m_h")) {
            pvtu_file << "   <PDataArray type=\"Float32\" Name=\"m_h\"/>\n";
        }
        if (print_info.at("level")) {
            pvtu_file << "   <PDataArray type=\"Int32\" Name=\"level\"/>\n";
        }
        if (print_info.at("neighbor_num")) {
            pvtu_file << "   <PDataArray type=\"UInt32\" Name=\"neighbor_num\"/>\n";
        }

        pvtu_file << "  </PPointData>\n"
                     "  <PPoints>\n"
                     "   <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n"
                     "  </PPoints>\n"
                  << "  <Piece Source=\"particle_step_" << n_out << ".vtu\"/>\n"
                     " </PUnstructuredGrid>\n"
                     "</VTKFile>\n";
    }

    void Output_pvd_file(int n_out) const {
        const std::filesystem::path filename = output_directory_ / "particle.pvd";
        std::vector<std::string> datasets;

        if (std::ifstream existing_file(filename); existing_file) {
            std::string line;
            while (std::getline(existing_file, line)) {
                if (line.find("<DataSet ") != std::string::npos) {
                    datasets.push_back(line);
                }
            }
        }

        const std::string dataset =
            "  <DataSet timestep=\"" + std::to_string(n_out) +
            "\" group=\"\" part=\"0\" file=\"particle_step_" +
            std::to_string(n_out) + ".pvtu\"/>";

        bool replaced = false;
        for (std::string& existing_dataset : datasets) {
            if (existing_dataset.find("timestep=\"" + std::to_string(n_out) +
                                     "\"") != std::string::npos) {
                existing_dataset = dataset;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            datasets.push_back(dataset);
        }

        std::ofstream pvd_file(filename, std::ios::trunc);
        if (!pvd_file) {
            return;
        }
        pvd_file << "<?xml version=\"1.0\"?>\n"
                    "<VTKFile type=\"Collection\" version=\"0.1\">\n"
                    " <Collection>\n";
        for (const std::string& dataset_entry : datasets) {
            pvd_file << dataset_entry << '\n';
        }
        pvd_file << " </Collection>\n</VTKFile>\n";
    }

private:
    std::filesystem::path output_directory_;

    static std::string data_array_header(const char* type, const char* name,
                                         std::uint32_t offset) {
        return "    <DataArray type=\"" + std::string(type) + "\" Name=\"" +
               name + "\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" +
               std::to_string(offset) + "\"/>\n";
    }

    static void append_block(std::ostringstream& stream, const void* data,
                             std::size_t byte_count, std::uint32_t& offset) {
        const auto block_size = static_cast<std::uint32_t>(byte_count);
        stream.write(reinterpret_cast<const char*>(&block_size), sizeof(block_size));
        if (byte_count != 0) {
            stream.write(static_cast<const char*>(data),
                         static_cast<std::streamsize>(byte_count));
        }
        offset += sizeof(block_size) + block_size;
    }

    static void append_ids(std::ostringstream& stream, std::size_t count,
                           std::uint32_t& offset) {
        std::vector<std::int32_t> ids(count);
        for (std::size_t index = 0; index < count; ++index) {
            ids[index] = static_cast<std::int32_t>(index);
        }
        append_block(stream, ids.data(), ids.size() * sizeof(ids[0]), offset);
    }

    template <typename T>
    static void append_values(std::ostringstream& stream,
                              const thrust::host_vector<T>& values,
                              std::size_t count, std::uint32_t& offset) {
        append_block(stream, values.data(), count * sizeof(T), offset);
    }

    static void append_coordinates(std::ostringstream& stream, const Points& points,
                                   std::uint32_t& offset) {
        std::vector<float> coordinates(points.particle_num * 3);
        for (std::size_t index = 0; index < points.particle_num; ++index) {
            coordinates[index * 3] = points.host_x_coord[index];
            coordinates[index * 3 + 1] = points.host_y_coord[index];
            coordinates[index * 3 + 2] = points.host_z_coord[index];
        }
        append_block(stream, coordinates.data(),
                     coordinates.size() * sizeof(coordinates[0]), offset);
    }

    static void append_connectivity(std::ostringstream& stream, std::size_t count,
                                    std::uint32_t& offset) {
        std::vector<std::int32_t> connectivity(count);
        for (std::size_t index = 0; index < count; ++index) {
            connectivity[index] = static_cast<std::int32_t>(index);
        }
        append_block(stream, connectivity.data(),
                     connectivity.size() * sizeof(connectivity[0]), offset);
    }

    static void append_offsets(std::ostringstream& stream, std::size_t count,
                               std::uint32_t& offset) {
        std::vector<std::int32_t> cell_offsets(count);
        for (std::size_t index = 0; index < count; ++index) {
            cell_offsets[index] = static_cast<std::int32_t>(index + 1);
        }
        append_block(stream, cell_offsets.data(),
                     cell_offsets.size() * sizeof(cell_offsets[0]), offset);
    }

    static void append_cell_types(std::ostringstream& stream, std::size_t count,
                                  std::uint32_t& offset) {
        const std::uint8_t vertex_cell_type = 1;
        std::vector<std::uint8_t> cell_types(count, vertex_cell_type);
        append_block(stream, cell_types.data(), cell_types.size(), offset);
    }
};

#endif