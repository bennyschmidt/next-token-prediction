/**
 * Vector
 * Extends Array with specified {DIMENSIONS}.
 */

const { DIMENSIONS = 144 } = process.env;
const RANGE_ERROR = 'RangeError: Invalid vector length.';

class Vector extends Array {
  static fromNull () {
    return this.from({ length: DIMENSIONS}).fill(0);
  }

  constructor () {
    super(...arguments);

    if (this.length !== DIMENSIONS) {
      throw RANGE_ERROR;
    }
  }
}

module.exports = Vector;
