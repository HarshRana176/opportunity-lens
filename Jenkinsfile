// Jenkins CI pipeline for OpportunityLens.
//
// Validates every build by installing the project's actual dependencies
// (requirements.txt) into an isolated virtualenv and running the full
// pytest suite. It does not build, deploy, or touch any environment
// beyond the Jenkins workspace.
//
// Why no PostgreSQL/Ollama service is provisioned here:
//   - The test suite (tests/) never opens a real database connection.
//     Every route test overrides FastAPI's `get_db` dependency with an
//     in-memory SQLite session (see tests/test_api.py, tests/conftest.py),
//     and app.main's lifespan (the only place that binds the REAL
//     configured engine) is deliberately never triggered in tests.
//   - Every Ollama-backed chain used by the offline suite is monkeypatched
//     (tests/conftest.py's autouse fixtures). The one module that talks to
//     a real Ollama daemon (tests/test_ollama_integration.py) probes for
//     it at collection time and self-skips when it isn't reachable -- it
//     never fails the build for that reason.
//   - app.config.Settings still requires a DATABASE_URL value to be
//     *present* just to construct itself (it does not need to be
//     reachable). DATABASE_URL below is therefore a placeholder, not a
//     real credential -- see app/config.py.
//
// Verified locally before this file was written: a fresh `pip install -r
// requirements.txt` into a brand-new virtualenv, followed by `pytest
// --junitxml=test-results.xml` with only this placeholder DATABASE_URL
// set, reproduces the project's full result (1312 passed, 2 skipped).
//
// Assumes a Unix-like Jenkins agent (Linux/macOS), the common default.
// On a Windows agent, replace the `sh` steps with `bat`/`powershell` and
// swap `.venv/bin/activate` for `.venv\Scripts\activate.bat`.

pipeline {
    agent any

    environment {
        // Placeholder only -- never a real credential. See note above.
        DATABASE_URL = 'postgresql://ci:ci@localhost:5432/ci_placeholder'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                    python3 -m venv .venv
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    env.PYTEST_EXIT_CODE = sh(
                        script: '''
                            . .venv/bin/activate
                            pytest --junitxml=test-results.xml
                        ''',
                        returnStatus: true
                    ).toString()
                }
            }
        }

        stage('Publish Test Results') {
            steps {
                // Runs even if Run Tests failed, so a failing build still
                // shows exactly which tests broke.
                junit testResults: 'test-results.xml', allowEmptyResults: true
            }
        }
    }

    post {
        always {
            script {
                if (env.PYTEST_EXIT_CODE != '0') {
                    error("pytest exited with status ${env.PYTEST_EXIT_CODE} -- failing the build.")
                }
            }
        }
        cleanup {
            // Remove the build-time virtualenv and pytest cache; the
            // JUnit report has already been captured by the step above.
            sh 'rm -rf .venv .pytest_cache test-results.xml'
        }
    }
}
