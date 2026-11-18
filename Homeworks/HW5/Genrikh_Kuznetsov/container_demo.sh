#!/bin/zsh
# Records the Milestone 5 container demo. Run after `make build`, from this folder:
#   zsh container_demo.sh > evidence/container_demo.txt
export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin"  # Docker Desktop CLI on macOS
run() { print -r -- "\$ $*"; eval "$@" 2>&1; print; }
wait_ready() { for i in {1..60}; do curl -sf 127.0.0.1:8000/health >/dev/null && return; sleep 0.5; done; echo "not ready"; }
print "# HW5 container demo - $(date '+%Y-%m-%d %H:%M:%S %Z') - $(sw_vers -productName) $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), Docker $(docker version --format '{{.Server.Version}}')"
print
run docker image ls mfe-hw5:local
print "## 1. Start the container (detached here so the transcript can continue; 'make run' runs it in the foreground)"
run docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
wait_ready
run docker ps --filter name=mfe-hw5 --format "'{{.Names}}  {{.Image}}  {{.Status}}  {{.Ports}}'"
print "## 2. Valid request (AAPL, 2026-08-31; HW4 recorded score 0.034079615723295494)"
run cat example_request.json
run curl -sS -i http://127.0.0.1:8000/predict -H "'Content-Type: application/json'" --data-binary @example_request.json
print "\n## 3. Invalid requests are rejected with 422"
run curl -sS -i http://127.0.0.1:8000/predict -H "'Content-Type: application/json'" --data "'$(python3 -c "import json;d=json.load(open('example_request.json'));d['momentum_21']='0.02';d['sector']='Crypto';print(json.dumps(d))")'"
print
run curl -sS -i http://127.0.0.1:8000/predict -H "'Content-Type: application/json'" --data "'$(python3 -c "import json;d=json.load(open('example_request.json'));d['momentum_21']=float('nan');d.pop('net_margin');print(json.dumps(d))")'"
print "\n## 4. Server log (one line per request; no input echoed)"
run docker logs mfe-hw5
print "## 5. Stop the container, start the same image again, request again"
run docker stop mfe-hw5
run docker ps -a --filter name=mfe-hw5 --format "'{{.Names}} {{.Status}}'"
run docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
wait_ready
run curl -s http://127.0.0.1:8000/health
print
run curl -sS -i http://127.0.0.1:8000/predict -H "'Content-Type: application/json'" --data-binary @example_request.json
print "\n## 6. Latency over real localhost HTTP to the container"
run .venv/bin/python measure_latency.py -n 200
run docker stop mfe-hw5
