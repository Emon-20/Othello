(() => {
  "use strict";

  const BOARD_SIZE = 8;
  const EMPTY = 0;
  const BLACK = 1;
  const WHITE = 2;

  const COL_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"];
  const DIRECTIONS = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1],
  ];

  // Classic Othello positional weights
  const POSITION_WEIGHTS = [
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
  ];

  const CORNER_ADJACENCY = [
    { corner: [0, 0], adjacent: [[0, 1], [1, 0], [1, 1]] },
    { corner: [0, 7], adjacent: [[0, 6], [1, 7], [1, 6]] },
    { corner: [7, 0], adjacent: [[6, 0], [7, 1], [6, 1]] },
    { corner: [7, 7], adjacent: [[6, 7], [7, 6], [6, 6]] },
  ];

  const STORAGE_KEY_PREFS = "othello_prefs_v1";
  const STORAGE_KEY_STATS = "othello_stats_v1";

  // =========================================================================
  // Pure Othello Rules Engine
  // =========================================================================

  function opponentOf(player) {
    return player === BLACK ? WHITE : BLACK;
  }

  function createInitialBoard() {
    const board = Array.from({ length: BOARD_SIZE }, () =>
      Array(BOARD_SIZE).fill(EMPTY)
    );
    board[3][3] = WHITE;
    board[3][4] = BLACK;
    board[4][3] = BLACK;
    board[4][4] = WHITE;
    return board;
  }

  function cloneBoard(board) {
    return board.map((row) => row.slice());
  }

  function isInsideBoard(r, c) {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
  }

  function getFlips(board, row, col, player) {
    if (!isInsideBoard(row, col) || board[row][col] !== EMPTY) {
      return [];
    }

    const opp = opponentOf(player);
    const allFlips = [];

    for (let i = 0; i < DIRECTIONS.length; i++) {
      const [dr, dc] = DIRECTIONS[i];
      let r = row + dr;
      let c = col + dc;
      const lineFlips = [];

      while (isInsideBoard(r, c) && board[r][c] === opp) {
        lineFlips.push([r, c]);
        r += dr;
        c += dc;
      }

      if (lineFlips.length > 0 && isInsideBoard(r, c) && board[r][c] === player) {
        for (let j = 0; j < lineFlips.length; j++) {
          allFlips.push(lineFlips[j]);
        }
      }
    }

    return allFlips;
  }

  function getValidMoves(board, player) {
    const moves = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (board[r][c] === EMPTY) {
          const flips = getFlips(board, r, c, player);
          if (flips.length > 0) {
            moves.push({ row: r, col: c, flips });
          }
        }
      }
    }
    return moves;
  }

  function applyMove(board, row, col, player) {
    const flips = getFlips(board, row, col, player);
    if (flips.length === 0) {
      return null;
    }

    const nextBoard = cloneBoard(board);
    nextBoard[row][col] = player;
    for (let i = 0; i < flips.length; i++) {
      const [fr, fc] = flips[i];
      nextBoard[fr][fc] = player;
    }

    return { nextBoard, flips };
  }

  function countDiscs(board) {
    let black = 0;
    let white = 0;
    let empty = 0;

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (board[r][c] === BLACK) black++;
        else if (board[r][c] === WHITE) white++;
        else empty++;
      }
    }

    return { black, white, empty };
  }

  function coordLabel(row, col) {
    return `${COL_LABELS[col]}${row + 1}`;
  }

  // =========================================================================
  // AI Engine (Easy, Medium, Hard Minimax with Alpha-Beta Pruning)
  // =========================================================================

  function getDynamicWeight(board, r, c, player) {
    let w = POSITION_WEIGHTS[r][c];
    // If a corner is already occupied by player, adjacent X/C squares are no longer risky
    for (let i = 0; i < CORNER_ADJACENCY.length; i++) {
      const { corner, adjacent } = CORNER_ADJACENCY[i];
      if (board[corner[0]][corner[1]] === player) {
        for (let j = 0; j < adjacent.length; j++) {
          if (adjacent[j][0] === r && adjacent[j][1] === c) {
            return 12;
          }
        }
      }
    }
    return w;
  }

  function evaluateMediumMove(board, move, player) {
    const baseWeight = getDynamicWeight(board, move.row, move.col, player);
    // Favor positional strength plus modest flip count
    return baseWeight + move.flips.length * 1.5;
  }

  function evaluateBoard(board, aiPlayer) {
    const opp = opponentOf(aiPlayer);
    const counts = countDiscs(board);
    const aiMoves = getValidMoves(board, aiPlayer).length;
    const oppMoves = getValidMoves(board, opp).length;

    // Terminal state check
    if (counts.empty === 0 || (aiMoves === 0 && oppMoves === 0)) {
      const diff = aiPlayer === BLACK
        ? counts.black - counts.white
        : counts.white - counts.black;
      if (diff > 0) return 100000 + diff;
      if (diff < 0) return -100000 + diff;
      return 0;
    }

    let positionalScore = 0;
    let cornerScore = 0;

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const cell = board[r][c];
        if (cell === aiPlayer) {
          positionalScore += getDynamicWeight(board, r, c, aiPlayer);
        } else if (cell === opp) {
          positionalScore -= getDynamicWeight(board, r, c, opp);
        }
      }
    }

    // Explicit corner ownership bonus
    const corners = [[0, 0], [0, 7], [7, 0], [7, 7]];
    for (let i = 0; i < corners.length; i++) {
      const [cr, cc] = corners[i];
      if (board[cr][cc] === aiPlayer) cornerScore += 30;
      else if (board[cr][cc] === opp) cornerScore -= 30;
    }

    // Mobility (normalized difference)
    let mobilityScore = 0;
    if (aiMoves + oppMoves > 0) {
      mobilityScore = (100 * (aiMoves - oppMoves)) / (aiMoves + oppMoves);
    }

    // Parity (disc count matters most in endgame)
    const aiDiscs = aiPlayer === BLACK ? counts.black : counts.white;
    const oppDiscs = aiPlayer === BLACK ? counts.white : counts.black;
    const parityScore = aiDiscs + oppDiscs > 0
      ? (100 * (aiDiscs - oppDiscs)) / (aiDiscs + oppDiscs)
      : 0;

    const totalPlaced = 64 - counts.empty;
    const parityWeight = totalPlaced > 48 ? 2.8 : totalPlaced > 30 ? 0.6 : 0.15;
    const mobilityWeight = totalPlaced > 50 ? 0.8 : 2.2;

    return (
      positionalScore * 1.0 +
      cornerScore * 1.5 +
      mobilityScore * mobilityWeight +
      parityScore * parityWeight
    );
  }

  function orderMoves(board, moves, player) {
    return moves.slice().sort((a, b) => {
      const wa = getDynamicWeight(board, a.row, a.col, player);
      const wb = getDynamicWeight(board, b.row, b.col, player);
      return wb - wa;
    });
  }

  function minimax(board, depth, alpha, beta, maximizingPlayer, aiPlayer, currentTurn) {
    const validMoves = getValidMoves(board, currentTurn);
    const oppTurn = opponentOf(currentTurn);

    if (depth === 0) {
      return { score: evaluateBoard(board, aiPlayer), move: null };
    }

    if (validMoves.length === 0) {
      const oppMoves = getValidMoves(board, oppTurn);
      if (oppMoves.length === 0) {
        return { score: evaluateBoard(board, aiPlayer), move: null };
      }
      // Pass turn
      return minimax(board, depth - 1, alpha, beta, !maximizingPlayer, aiPlayer, oppTurn);
    }

    const ordered = orderMoves(board, validMoves, currentTurn);
    let bestMove = ordered[0];

    if (maximizingPlayer) {
      let maxEval = -Infinity;
      for (let i = 0; i < ordered.length; i++) {
        const m = ordered[i];
        const { nextBoard } = applyMove(board, m.row, m.col, currentTurn);
        const result = minimax(nextBoard, depth - 1, alpha, beta, false, aiPlayer, oppTurn);
        if (result.score > maxEval) {
          maxEval = result.score;
          bestMove = m;
        }
        if (maxEval > alpha) alpha = maxEval;
        if (beta <= alpha) break;
      }
      return { score: maxEval, move: bestMove };
    } else {
      let minEval = Infinity;
      for (let i = 0; i < ordered.length; i++) {
        const m = ordered[i];
        const { nextBoard } = applyMove(board, m.row, m.col, currentTurn);
        const result = minimax(nextBoard, depth - 1, alpha, beta, true, aiPlayer, oppTurn);
        if (result.score < minEval) {
          minEval = result.score;
          bestMove = m;
        }
        if (minEval < beta) beta = minEval;
        if (beta <= alpha) break;
      }
      return { score: minEval, move: bestMove };
    }
  }

  function chooseAIMove(board, aiPlayer, difficulty) {
    const validMoves = getValidMoves(board, aiPlayer);
    if (validMoves.length === 0) return null;

    if (difficulty === "easy") {
      const idx = Math.floor(Math.random() * validMoves.length);
      return validMoves[idx];
    }

    if (difficulty === "medium") {
      let bestScore = -Infinity;
      let bestMoves = [];
      for (let i = 0; i < validMoves.length; i++) {
        const m = validMoves[i];
        const score = evaluateMediumMove(board, m, aiPlayer);
        if (score > bestScore) {
          bestScore = score;
          bestMoves = [m];
        } else if (score === bestScore) {
          bestMoves.push(m);
        }
      }
      return bestMoves[Math.floor(Math.random() * bestMoves.length)];
    }

    // Hard: Minimax with Alpha-Beta pruning (depth 5, deeper in late endgame)
    const { empty } = countDiscs(board);
    const depth = empty <= 10 ? 7 : 5;
    const result = minimax(board, depth, -Infinity, Infinity, true, aiPlayer, aiPlayer);
    return result.move || validMoves[0];
  }

  // Export pure engine for automated Node tests if running outside browser
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      BOARD_SIZE,
      EMPTY,
      BLACK,
      WHITE,
      opponentOf,
      createInitialBoard,
      getFlips,
      getValidMoves,
      applyMove,
      countDiscs,
      chooseAIMove,
      coordLabel,
    };
    return;
  }

  // =========================================================================
  // Web Audio Synthesizer (Subtle Sound Effects)
  // =========================================================================

  let audioCtx = null;

  function ensureAudioContext() {
    if (!audioCtx && (window.AudioContext || window.webkitAudioContext)) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctx();
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
    return audioCtx;
  }

  function playTone(freq, duration, type = "sine", volume = 0.08, delay = 0) {
    if (!state.prefs.sound) return;
    const ctx = ensureAudioContext();
    if (!ctx) return;

    const now = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, now);
    gain.gain.setValueAtTime(volume, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + duration);
  }

  function playPlaceSound(flipCount) {
    playTone(210, 0.07, "triangle", 0.11, 0);
    const clicks = Math.min(flipCount, 4);
    for (let i = 0; i < clicks; i++) {
      playTone(340 + i * 45, 0.05, "sine", 0.045, 0.04 + i * 0.04);
    }
  }

  function playPassSound() {
    playTone(330, 0.12, "sine", 0.06, 0);
    playTone(262, 0.16, "sine", 0.06, 0.11);
  }

  function playGameOverSound(isWin) {
    if (isWin) {
      playTone(523.25, 0.14, "triangle", 0.08, 0);
      playTone(659.25, 0.14, "triangle", 0.08, 0.12);
      playTone(783.99, 0.25, "triangle", 0.09, 0.24);
    } else {
      playTone(392.0, 0.16, "sine", 0.07, 0);
      playTone(349.23, 0.22, "sine", 0.07, 0.15);
    }
  }

  // =========================================================================
  // Persistence (localStorage)
  // =========================================================================

  function loadStorage(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? { ...fallback, ...JSON.parse(raw) } : { ...fallback };
    } catch {
      return { ...fallback };
    }
  }

  function saveStorage(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Ignore storage quota / privacy mode errors
    }
  }

  const DEFAULT_PREFS = {
    mode: "ai", // "ai" | "friend"
    difficulty: "medium", // "easy" | "medium" | "hard"
    humanColor: BLACK,
    hints: true,
    sound: true,
    animations: true,
  };

  const DEFAULT_STATS = {
    aiWins: 0,
    aiLosses: 0,
    aiTies: 0,
    friendBlackWins: 0,
    friendWhiteWins: 0,
    friendTies: 0,
  };

  // =========================================================================
  // Application State & DOM References
  // =========================================================================

  const state = {
    board: createInitialBoard(),
    currentPlayer: BLACK,
    validMoves: [],
    lastMove: null, // { row, col }
    moveLog: [], // [{ num, player, coord, flips, isPass }]
    history: [], // snapshots for undo
    isGameOver: false,
    isAIThinking: false,
    aiTimerId: null,
    gameSessionId: 0,
    prefs: loadStorage(STORAGE_KEY_PREFS, DEFAULT_PREFS),
    stats: loadStorage(STORAGE_KEY_STATS, DEFAULT_STATS),
  };

  const dom = {
    board: document.getElementById("board"),
    scoreBlack: document.getElementById("score-black"),
    scoreWhite: document.getElementById("score-white"),
    cardBlack: document.getElementById("card-black"),
    cardWhite: document.getElementById("card-white"),
    labelBlack: document.getElementById("label-black"),
    labelWhite: document.getElementById("label-white"),
    badgeBlack: document.getElementById("badge-black"),
    badgeWhite: document.getElementById("badge-white"),
    statusBanner: document.getElementById("status-banner"),
    modeSelect: document.getElementById("mode-select"),
    difficultySelect: document.getElementById("difficulty-select"),
    difficultyGroup: document.getElementById("difficulty-group"),
    colorSelect: document.getElementById("color-select"),
    colorGroup: document.getElementById("color-group"),
    btnUndo: document.getElementById("btn-undo"),
    btnNewGame: document.getElementById("btn-new-game"),
    btnResetStats: document.getElementById("btn-reset-stats"),
    toggleHints: document.getElementById("toggle-hints"),
    toggleSound: document.getElementById("toggle-sound"),
    toggleAnimations: document.getElementById("toggle-animations"),
    moveHistory: document.getElementById("move-history"),
    moveCount: document.getElementById("move-count"),
    statWins: document.getElementById("stat-wins"),
    statLosses: document.getElementById("stat-losses"),
    statTies: document.getElementById("stat-ties"),
    statsGrid: document.getElementById("stats-grid"),
    modal: document.getElementById("game-over-modal"),
    modalTrophy: document.getElementById("modal-trophy"),
    modalTitle: document.getElementById("modal-title"),
    modalSubtitle: document.getElementById("modal-subtitle"),
    modalBlackScore: document.getElementById("modal-black-score"),
    modalWhiteScore: document.getElementById("modal-white-score"),
    modalPlayAgain: document.getElementById("modal-play-again"),
    modalClose: document.getElementById("modal-close"),
  };

  const cellElements = []; // 8x8 array of <button> elements

  // =========================================================================
  // Board Initialization & Rendering
  // =========================================================================

  function buildBoardDOM() {
    dom.board.innerHTML = "";
    const starMap = {
      "1,1": "tl",
      "1,5": "tr",
      "5,1": "bl",
      "5,5": "br",
    };

    for (let r = 0; r < BOARD_SIZE; r++) {
      cellElements[r] = [];
      for (let c = 0; c < BOARD_SIZE; c++) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "cell";
        cell.dataset.row = String(r);
        cell.dataset.col = String(c);
        cell.setAttribute("role", "gridcell");

        const star = starMap[`${r},${c}`];
        if (star) {
          cell.dataset.star = star;
        }

        cell.addEventListener("click", () => handleCellClick(r, c));
        dom.board.appendChild(cell);
        cellElements[r][c] = cell;
      }
    }
  }

  function renderBoard(placedCoord = null) {
    const validSet = new Map();
    const canHumanAct = !state.isGameOver && !state.isAIThinking && isHumanTurn();

    if (canHumanAct) {
      for (let i = 0; i < state.validMoves.length; i++) {
        const m = state.validMoves[i];
        validSet.set(`${m.row},${m.col}`, m);
      }
    }

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const cell = cellElements[r][c];
        const piece = state.board[r][c];
        const key = `${r},${c}`;
        const isValid = validSet.has(key);
        const isLast = state.lastMove && state.lastMove.row === r && state.lastMove.col === c;

        cell.classList.toggle("valid-move", isValid);
        cell.classList.toggle("last-move", Boolean(isLast));
        cell.disabled = !isValid;

        // Accessibility label
        const label = coordLabel(r, c);
        if (piece === BLACK) {
          cell.setAttribute("aria-label", `${label}, Black disc`);
        } else if (piece === WHITE) {
          cell.setAttribute("aria-label", `${label}, White disc`);
        } else if (isValid) {
          cell.setAttribute("aria-label", `${label}, valid move`);
        } else {
          cell.setAttribute("aria-label", `${label}, empty`);
        }

        let discEl = cell.querySelector(".disc");
        let hintEl = cell.querySelector(".hint-dot");

        if (piece === EMPTY) {
          if (discEl) discEl.remove();
          if (isValid && !hintEl) {
            hintEl = document.createElement("span");
            hintEl.className = "hint-dot";
            hintEl.setAttribute("aria-hidden", "true");
            cell.appendChild(hintEl);
          } else if (!isValid && hintEl) {
            hintEl.remove();
          }
        } else {
          if (hintEl) hintEl.remove();

          const colorClass = piece === BLACK ? "is-black" : "is-white";
          const justPlaced = placedCoord && placedCoord.row === r && placedCoord.col === c;

          if (!discEl) {
            discEl = document.createElement("div");
            discEl.className = `disc ${colorClass}`;
            discEl.setAttribute("aria-hidden", "true");

            const blackFace = document.createElement("div");
            blackFace.className = "disc-face black";
            const whiteFace = document.createElement("div");
            whiteFace.className = "disc-face white";

            discEl.appendChild(blackFace);
            discEl.appendChild(whiteFace);
            cell.appendChild(discEl);

            if (justPlaced) {
              discEl.style.setProperty(
                "--disc-rot",
                piece === BLACK ? "rotateY(0deg)" : "rotateY(180deg)"
              );
              discEl.classList.add("placing");
              discEl.addEventListener(
                "animationend",
                () => discEl.classList.remove("placing"),
                { once: true }
              );
            }
          } else {
            // Reuse existing disc element so CSS 3D flip transition animates smoothly!
            discEl.classList.toggle("is-black", piece === BLACK);
            discEl.classList.toggle("is-white", piece === WHITE);
          }
        }
      }
    }
  }

  function renderScoreboardAndStatus(customBanner = null, bannerClass = "") {
    const counts = countDiscs(state.board);
    dom.scoreBlack.textContent = String(counts.black);
    dom.scoreWhite.textContent = String(counts.white);

    const isAI = state.prefs.mode === "ai";
    const humanColor = Number(state.prefs.humanColor);

    if (isAI) {
      dom.labelBlack.textContent = humanColor === BLACK ? "Black (You)" : "Black (AI)";
      dom.labelWhite.textContent = humanColor === WHITE ? "White (You)" : "White (AI)";
    } else {
      dom.labelBlack.textContent = "Black (P1)";
      dom.labelWhite.textContent = "White (P2)";
    }

    dom.cardBlack.classList.toggle("active-turn", !state.isGameOver && state.currentPlayer === BLACK);
    dom.cardWhite.classList.toggle("active-turn", !state.isGameOver && state.currentPlayer === WHITE);

    if (state.isGameOver) {
      dom.badgeBlack.textContent = "Final";
      dom.badgeWhite.textContent = "Final";
    } else {
      const blackActive = state.currentPlayer === BLACK;
      dom.badgeBlack.textContent = blackActive
        ? (state.isAIThinking && isAITurn() ? "Thinking…" : "Active")
        : "Waiting";
      dom.badgeWhite.textContent = !blackActive
        ? (state.isAIThinking && isAITurn() ? "Thinking…" : "Active")
        : "Waiting";
    }

    dom.statusBanner.className = "status-banner" + (bannerClass ? ` ${bannerClass}` : "");
    if (customBanner) {
      dom.statusBanner.textContent = customBanner;
    } else if (state.isGameOver) {
      if (counts.black > counts.white) {
        dom.statusBanner.textContent = `Game Over — Black wins ${counts.black} to ${counts.white}!`;
      } else if (counts.white > counts.black) {
        dom.statusBanner.textContent = `Game Over — White wins ${counts.white} to ${counts.black}!`;
      } else {
        dom.statusBanner.textContent = `Game Over — It's a ${counts.black}–${counts.white} tie!`;
      }
      dom.statusBanner.classList.add("gameover-alert");
    } else if (state.isAIThinking) {
      const aiName = state.currentPlayer === BLACK ? "Black AI" : "White AI";
      dom.statusBanner.textContent = `${aiName} is thinking…`;
    } else {
      const turnName = state.currentPlayer === BLACK ? "Black" : "White";
      const movesCount = state.validMoves.length;
      dom.statusBanner.textContent = `${turnName}'s turn (${movesCount} legal ${movesCount === 1 ? "move" : "moves"})`;
    }

    dom.btnUndo.disabled = !canUndo();
  }

  function renderMoveHistory() {
    const entries = state.moveLog;
    const playedMoves = entries.filter((e) => !e.isPass).length;
    dom.moveCount.textContent = `${playedMoves} ${playedMoves === 1 ? "move" : "moves"}`;

    if (entries.length === 0) {
      dom.moveHistory.innerHTML = `<li class="history-empty">No moves played yet.</li>`;
      return;
    }

    dom.moveHistory.innerHTML = entries
      .map((entry) => {
        const discClass = entry.player === BLACK ? "black" : "white";
        if (entry.isPass) {
          return `
            <li class="history-item is-pass">
              <span class="history-num">#${entry.num}</span>
              <span class="history-disc ${discClass}"></span>
              <span class="history-coord">PASS</span>
              <span class="history-flips">No legal moves</span>
            </li>
          `;
        }
        return `
          <li class="history-item">
            <span class="history-num">#${entry.num}</span>
            <span class="history-disc ${discClass}"></span>
            <span class="history-coord">${entry.coord}</span>
            <span class="history-flips">+${entry.flips} ${entry.flips === 1 ? "flip" : "flips"}</span>
          </li>
        `;
      })
      .join("");

    dom.moveHistory.scrollTop = dom.moveHistory.scrollHeight;
  }

  function renderStats() {
    const isAI = state.prefs.mode === "ai";
    const labels = dom.statsGrid.querySelectorAll(".stat-label");

    if (isAI) {
      dom.statWins.textContent = String(state.stats.aiWins);
      dom.statLosses.textContent = String(state.stats.aiLosses);
      dom.statTies.textContent = String(state.stats.aiTies);
      labels[0].textContent = "AI Wins";
      labels[1].textContent = "AI Losses";
      labels[2].textContent = "Ties";
    } else {
      dom.statWins.textContent = String(state.stats.friendBlackWins);
      dom.statLosses.textContent = String(state.stats.friendWhiteWins);
      dom.statTies.textContent = String(state.stats.friendTies);
      labels[0].textContent = "Black Wins";
      labels[1].textContent = "White Wins";
      labels[2].textContent = "Ties";
    }
  }

  function syncPreferenceUI() {
    dom.modeSelect.value = state.prefs.mode;
    dom.difficultySelect.value = state.prefs.difficulty;
    dom.colorSelect.value = String(state.prefs.humanColor);

    const isAI = state.prefs.mode === "ai";
    dom.difficultyGroup.classList.toggle("hidden", !isAI);
    dom.colorGroup.classList.toggle("hidden", !isAI);

    dom.toggleHints.classList.toggle("active", state.prefs.hints);
    dom.toggleHints.setAttribute("aria-pressed", String(state.prefs.hints));
    document.body.classList.toggle("hints-hidden", !state.prefs.hints);

    dom.toggleSound.classList.toggle("active", state.prefs.sound);
    dom.toggleSound.setAttribute("aria-pressed", String(state.prefs.sound));

    dom.toggleAnimations.classList.toggle("active", state.prefs.animations);
    dom.toggleAnimations.setAttribute("aria-pressed", String(state.prefs.animations));
    document.body.classList.toggle("no-animations", !state.prefs.animations);
  }

  // =========================================================================
  // Game Flow & Turn Management
  // =========================================================================

  function isAITurn() {
    return (
      state.prefs.mode === "ai" &&
      state.currentPlayer !== Number(state.prefs.humanColor)
    );
  }

  function isHumanTurn() {
    return !isAITurn();
  }

  function cancelPendingAI() {
    if (state.aiTimerId !== null) {
      clearTimeout(state.aiTimerId);
      state.aiTimerId = null;
    }
    state.isAIThinking = false;
  }

  function startNewGame() {
    cancelPendingAI();
    state.gameSessionId += 1;
    state.board = createInitialBoard();
    state.currentPlayer = BLACK;
    state.validMoves = getValidMoves(state.board, BLACK);
    state.lastMove = null;
    state.moveLog = [];
    state.history = [];
    state.isGameOver = false;

    if (dom.modal.open) {
      dom.modal.close();
    }

    renderBoard();
    renderMoveHistory();
    renderScoreboardAndStatus("Black moves first. Select a highlighted square.");
    renderStats();

    if (isAITurn()) {
      scheduleAIMove();
    }
  }

  function handleCellClick(row, col) {
    if (state.isGameOver || state.isAIThinking || !isHumanTurn()) return;
    executeMove(row, col);
  }

  function executeMove(row, col) {
    const result = applyMove(state.board, row, col, state.currentPlayer);
    if (!result) return;

    // Save snapshot for Undo before mutating state
    state.history.push({
      board: cloneBoard(state.board),
      currentPlayer: state.currentPlayer,
      lastMove: state.lastMove ? { ...state.lastMove } : null,
      moveLogLength: state.moveLog.length,
    });

    const movingPlayer = state.currentPlayer;
    state.board = result.nextBoard;
    state.lastMove = { row, col };
    state.moveLog.push({
      num: state.moveLog.length + 1,
      player: movingPlayer,
      coord: coordLabel(row, col),
      flips: result.flips.length,
      isPass: false,
    });

    playPlaceSound(result.flips.length);
    advanceTurn(movingPlayer, { row, col });
  }

  function advanceTurn(lastPlayer, placedCoord) {
    const nextPlayer = opponentOf(lastPlayer);
    const nextMoves = getValidMoves(state.board, nextPlayer);

    if (nextMoves.length > 0) {
      state.currentPlayer = nextPlayer;
      state.validMoves = nextMoves;
      renderBoard(placedCoord);
      renderMoveHistory();
      renderScoreboardAndStatus();

      if (isAITurn()) {
        scheduleAIMove();
      }
      return;
    }

    // Next player has no legal moves — check if lastPlayer can move
    const prevPlayerMoves = getValidMoves(state.board, lastPlayer);
    if (prevPlayerMoves.length === 0) {
      // Neither player can move -> Game Over
      finishGame(placedCoord);
      return;
    }

    // Automatic turn pass!
    const skippedName = nextPlayer === BLACK ? "Black" : "White";
    const activeName = lastPlayer === BLACK ? "Black" : "White";

    state.moveLog.push({
      num: state.moveLog.length + 1,
      player: nextPlayer,
      coord: "PASS",
      flips: 0,
      isPass: true,
    });

    state.currentPlayer = lastPlayer;
    state.validMoves = prevPlayerMoves;

    playPassSound();
    renderBoard(placedCoord);
    renderMoveHistory();
    renderScoreboardAndStatus(
      `${skippedName} has no legal moves — Turn passes to ${activeName}!`,
      "pass-alert"
    );

    if (isAITurn()) {
      scheduleAIMove();
    }
  }

  function scheduleAIMove() {
    cancelPendingAI();
    state.isAIThinking = true;
    const sessionAtSchedule = state.gameSessionId;
    renderBoard();
    renderScoreboardAndStatus();

    const delay = state.prefs.animations ? 420 : 220;
    state.aiTimerId = setTimeout(() => {
      state.aiTimerId = null;
      if (state.gameSessionId !== sessionAtSchedule || state.isGameOver) {
        return;
      }

      state.isAIThinking = false;
      const move = chooseAIMove(
        state.board,
        state.currentPlayer,
        state.prefs.difficulty
      );

      if (move) {
        executeMove(move.row, move.col);
      }
    }, delay);
  }

  function canUndo() {
    if (state.history.length === 0) return false;
    if (state.prefs.mode === "ai") {
      // Must have at least one human move in history to revert back to
      const humanColor = Number(state.prefs.humanColor);
      return state.history.some((snap) => snap.currentPlayer === humanColor);
    }
    return true;
  }

  function handleUndo() {
    if (!canUndo()) return;
    cancelPendingAI();
    state.gameSessionId += 1;

    if (dom.modal.open) {
      dom.modal.close();
    }

    let snapshot;
    if (state.prefs.mode === "ai") {
      const humanColor = Number(state.prefs.humanColor);
      // Pop back until we reach a snapshot where it was the human's turn
      while (state.history.length > 0) {
        snapshot = state.history.pop();
        if (snapshot.currentPlayer === humanColor) {
          break;
        }
      }
    } else {
      snapshot = state.history.pop();
    }

    if (!snapshot) return;

    state.board = cloneBoard(snapshot.board);
    state.currentPlayer = snapshot.currentPlayer;
    state.lastMove = snapshot.lastMove;
    state.moveLog = state.moveLog.slice(0, snapshot.moveLogLength);
    state.isGameOver = false;
    state.validMoves = getValidMoves(state.board, state.currentPlayer);

    renderBoard();
    renderMoveHistory();
    renderScoreboardAndStatus("Move undone.");
  }

  function finishGame(placedCoord) {
    state.isGameOver = true;
    state.validMoves = [];
    cancelPendingAI();

    renderBoard(placedCoord);
    renderMoveHistory();

    const counts = countDiscs(state.board);
    const isAI = state.prefs.mode === "ai";
    const humanColor = Number(state.prefs.humanColor);

    let title = "Game Over";
    let subtitle = "";
    let trophy = "🏆";
    let isWin = true;

    if (counts.black === counts.white) {
      title = "It's a Tie!";
      subtitle = `Both players finished with ${counts.black} discs.`;
      trophy = "🤝";
      if (isAI) state.stats.aiTies += 1;
      else state.stats.friendTies += 1;
    } else {
      const winner = counts.black > counts.white ? BLACK : WHITE;
      const winnerName = winner === BLACK ? "Black" : "White";

      if (isAI) {
        if (winner === humanColor) {
          title = "Victory!";
          subtitle = `You (${winnerName}) defeated the ${state.prefs.difficulty} AI!`;
          trophy = "🏆";
          state.stats.aiWins += 1;
          isWin = true;
        } else {
          title = "AI Wins";
          subtitle = `The ${state.prefs.difficulty} AI (${winnerName}) took the match.`;
          trophy = "🤖";
          state.stats.aiLosses += 1;
          isWin = false;
        }
      } else {
        title = `${winnerName} Wins!`;
        subtitle = `${winnerName} leads ${Math.max(counts.black, counts.white)} to ${Math.min(counts.black, counts.white)}.`;
        trophy = "🏆";
        if (winner === BLACK) state.stats.friendBlackWins += 1;
        else state.stats.friendWhiteWins += 1;
      }
    }

    saveStorage(STORAGE_KEY_STATS, state.stats);
    renderStats();
    renderScoreboardAndStatus();
    playGameOverSound(isWin);

    dom.modalTrophy.textContent = trophy;
    dom.modalTitle.textContent = title;
    dom.modalSubtitle.textContent = subtitle;
    dom.modalBlackScore.textContent = String(counts.black);
    dom.modalWhiteScore.textContent = String(counts.white);

    if (typeof dom.modal.showModal === "function") {
      dom.modal.showModal();
    } else {
      dom.modal.setAttribute("open", "");
    }
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================

  function bindEvents() {
    dom.btnNewGame.addEventListener("click", startNewGame);
    dom.btnUndo.addEventListener("click", handleUndo);

    dom.modeSelect.addEventListener("change", (e) => {
      state.prefs.mode = e.target.value;
      saveStorage(STORAGE_KEY_PREFS, state.prefs);
      syncPreferenceUI();
      startNewGame();
    });

    dom.difficultySelect.addEventListener("change", (e) => {
      state.prefs.difficulty = e.target.value;
      saveStorage(STORAGE_KEY_PREFS, state.prefs);
      startNewGame();
    });

    dom.colorSelect.addEventListener("change", (e) => {
      state.prefs.humanColor = Number(e.target.value);
      saveStorage(STORAGE_KEY_PREFS, state.prefs);
      startNewGame();
    });

    dom.toggleHints.addEventListener("click", () => {
      state.prefs.hints = !state.prefs.hints;
      saveStorage(STORAGE_KEY_PREFS, state.prefs);
      syncPreferenceUI();
    });

    dom.toggleSound.addEventListener("click", () => {
      state.prefs.sound = !state.prefs.sound;
      saveStorage(STORAGE_KEY_PREFS, state.prefs);
      syncPreferenceUI();
    });

    dom.toggleAnimations.addEventListener("click", () => {
      state.prefs.animations = !state.prefs.animations;
      saveStorage(STORAGE_KEY_PREFS, state.prefs);
      syncPreferenceUI();
    });

    dom.btnResetStats.addEventListener("click", () => {
      state.stats = { ...DEFAULT_STATS };
      saveStorage(STORAGE_KEY_STATS, state.stats);
      renderStats();
    });

    dom.modalPlayAgain.addEventListener("click", startNewGame);
    dom.modalClose.addEventListener("click", () => {
      if (dom.modal.open) dom.modal.close();
      else dom.modal.removeAttribute("open");
    });

    // Light dismiss on backdrop click
    dom.modal.addEventListener("click", (e) => {
      if (e.target === dom.modal) {
        dom.modal.close();
      }
    });
  }

  // =========================================================================
  // Bootstrap
  // =========================================================================

  function init() {
    buildBoardDOM();
    syncPreferenceUI();
    bindEvents();
    startNewGame();
  }

  init();
})();

