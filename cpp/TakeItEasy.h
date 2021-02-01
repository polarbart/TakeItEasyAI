#ifndef CPP_TAKEITEASY_H
#define CPP_TAKEITEASY_H

#include <iostream>
#include <random>
#include <optional>

#define STATE_SIZE (19*3*3)
#define INVALID_SPACE 19
#define INVALID_PIECE 27

extern const std::int8_t numbers_on_pieces[28][3];
extern const std::int8_t spaces_on_lines[3][5][5];
extern const std::int8_t lines_on_space[19][3];
extern const std::int8_t numbers_for_dirs[3][3];

class TakeItEasy {

public:
    explicit TakeItEasy(std::optional<std::int32_t> seed);
    void reset();
    void setNextPiece(std::int8_t piece);
    void swapCurrentPieceWith(std::int8_t swp);
    void place(std::int8_t pos);
    void undo();

    [[nodiscard]] std::int16_t computeScore() const; // maximal score change is 307 > 255
    [[nodiscard]] std::int8_t computeScoreDelta(std::int8_t pos) const; // maximal score change is (9+8+7)*5 = 120 < 127

    void encode(std::int8_t* buf) const;

    // The round in which the game is. Also equals the number of pieces already on the board. Therefore between 0 and 19
    std::int8_t step;
    std::int8_t board[20]; // + invalid pos
    std::int8_t subset[27];
    std::int8_t last_positions[19];

private:
    std::int8_t getRandInt(std::int8_t m);
    std::mt19937 rng;
};


#endif //CPP_TAKEITEASY_H
