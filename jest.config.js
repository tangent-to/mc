export default {
  testEnvironment: 'node',
  transform: {},
  testMatch: [
    '**/tests/**/*.test.js'
  ],
  collectCoverageFrom: [
    'src/**/*.js'
  ],
  coverageDirectory: 'coverage',
  verbose: true,
  testTimeout: 30000
};
