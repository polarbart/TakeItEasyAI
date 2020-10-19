//
// Created by Julius on 19.05.2020.
//
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <optional>
#include "TakeItEasy.h"

namespace py = pybind11;

std::vector<std::uint8_t> empty_tiles(const TakeItEasy &t) {
    std::vector<std::uint8_t> ret;
    for (std::int8_t i = 0; i < 19; ++i)
        if (t.board[i] == INVALID_PIECE)
            ret.push_back(i);
    return ret;
}

std::vector<py::tuple> remaining_pieces(const TakeItEasy &t) {
    std::vector<py::tuple> ret;
    for (std::int8_t i = t.step; i < 27; ++i)
        ret.push_back(py::make_tuple(
                numbers_on_pieces[t.subset[i]][0],
                numbers_on_pieces[t.subset[i]][1],
                numbers_on_pieces[t.subset[i]][2]
            )
        );
    return ret;
}

std::optional<py::tuple> get_piece_at(const TakeItEasy &t, std::int8_t i) {
    if (t.board[i] == INVALID_PIECE)
        return std::nullopt;
    return std::optional<py::tuple>(py::make_tuple(
            numbers_on_pieces[t.board[i]][0],
            numbers_on_pieces[t.board[i]][1],
            numbers_on_pieces[t.board[i]][2]
    ));
}

std::optional<py::tuple> next_piece(const TakeItEasy &t) {
    if (t.step >= 19)
        return std::nullopt;
    return std::optional<py::tuple>(py::make_tuple(
            numbers_on_pieces[t.subset[t.step]][0],
            numbers_on_pieces[t.subset[t.step]][1],
            numbers_on_pieces[t.subset[t.step]][2]
    ));
}

py::array_t<std::float_t , py::array::c_style> encode(const TakeItEasy &t) {

    auto *data = new std::float_t[STATE_SIZE];
    t.encode(data);

    py::capsule free(data, [](void *d) {delete[] reinterpret_cast<std::float_t*>(d);});
    return py::array_t<std::float_t , py::array::c_style>(
            {STATE_SIZE},
            {sizeof(std::float_t)},
            data,
            free
    );
}


PYBIND11_MODULE(TakeItEasyC, m) {
    auto tie = py::class_<TakeItEasy>(m, "TakeItEasy");
    tie.def(py::init<std::int32_t>(), py::arg("seed") = -1);
    tie.def("reset", &TakeItEasy::reset);
    tie.def("place", &TakeItEasy::place);
    tie.def("undo", &TakeItEasy::undo);
    tie.def("compute_score", &TakeItEasy::computeScore);
    tie.def("encode", &encode);
    tie.def("set_next_piece", &TakeItEasy::setNextPiece);
    tie.def("swap_current_piece_with", &TakeItEasy::swapCurrentPieceWith);

    tie.def_readonly("step", &TakeItEasy::step);
    tie.def_property_readonly("empty_tiles", &empty_tiles);
    tie.def_property_readonly("remaining_pieces", &remaining_pieces);
    tie.def("get_piece_at", &get_piece_at);
    tie.def_property_readonly("next_piece", &next_piece);

    auto btie = py::class_<TakeItEasy::BatchedTakeItEasy, std::shared_ptr<TakeItEasy::BatchedTakeItEasy>>(m, "BatchedTakeItEasy");
    btie.def(py::init<std::int32_t, std::int32_t>(), py::arg("nGames"), py::arg("seed") = -1);
    btie.def("place", &TakeItEasy::BatchedTakeItEasy::place);
    btie.def("reset", &TakeItEasy::BatchedTakeItEasy::reset);
    btie.def("compute_encodings", &TakeItEasy::BatchedTakeItEasy::computeEncodings, py::arg("iterOverRemainingPieces") = true);
    btie.def("compute_scores", &TakeItEasy::BatchedTakeItEasy::computeScores);

    btie.def(py::pickle(
            [](const TakeItEasy::BatchedTakeItEasy &t) {
                return py::make_tuple(t.games.size());
            },
            [](const py::tuple& s) {
                return TakeItEasy::BatchedTakeItEasy(s[0].cast<std::int32_t>());
            }
    ));
}



